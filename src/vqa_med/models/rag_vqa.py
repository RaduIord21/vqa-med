"""
RAG-enhanced VQA model.
Combines visual question answering with retrieved medical knowledge.
"""
import torch
import torch.nn as nn
from transformers import ViTModel, AutoModel
from typing import Optional, List

from ..retrieval import MedicalKnowledgeBase, MedicalRetriever


class RAGVQAModel(nn.Module):
    """
    VQA model enhanced with Retrieval-Augmented Generation.
    
    Architecture:
    1. Retrieve relevant medical knowledge based on question
    2. Encode: Image + Question + Retrieved Context
    3. Cross-attention fusion
    4. Answer prediction
    """
    
    def __init__(
        self,
        num_classes: int,
        knowledge_base_path: str,
        vision_model_name: str = "google/vit-base-patch16-224",
        text_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        hidden_dim: int = 768,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        top_k_docs: int = 3,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.top_k_docs = top_k_docs
        
        # Load knowledge base for retrieval
        print(f"Loading knowledge base from: {knowledge_base_path}")
        self.knowledge_base = MedicalKnowledgeBase.load(knowledge_base_path)
        self.retriever = MedicalRetriever(self.knowledge_base, top_k=top_k_docs)
        
        # Vision Encoder
        print(f"Loading vision model: {vision_model_name}")
        self.vision_encoder = ViTModel.from_pretrained(vision_model_name)
        self.vision_dim = self.vision_encoder.config.hidden_size
        
        # Text Encoder (for question + context)
        print(f"Loading text model: {text_model_name}")
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.text_dim = self.text_encoder.config.hidden_size
        
        # Project vision to text dimension
        if self.vision_dim != self.text_dim:
            self.vision_proj = nn.Linear(self.vision_dim, self.text_dim)
        else:
            self.vision_proj = nn.Identity()
        
        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.text_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(self.text_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        print(f"RAG-VQA Model initialized:")
        print(f"  Vision dim: {self.vision_dim}")
        print(f"  Text dim: {self.text_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Top-K docs: {top_k_docs}")
        print(f"  Num classes: {num_classes}")
    
    def retrieve_knowledge(self, questions: List[str]) -> List[str]:
        """
        Retrieve relevant medical knowledge for batch of questions.
        
        Args:
            questions: List of question strings
            
        Returns:
            List of formatted context strings
        """
        contexts = []
        for question in questions:
            retrieved_docs = self.retriever.retrieve(question)
            context = self.retriever.format_context(retrieved_docs)
            contexts.append(context)
        return contexts
    
    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        context_input_ids: Optional[torch.Tensor] = None,
        context_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with RAG.
        
        Args:
            image: [batch_size, 3, H, W]
            input_ids: Question tokens [batch_size, seq_len]
            attention_mask: Question mask [batch_size, seq_len]
            context_input_ids: Retrieved context tokens [batch_size, context_len]
            context_attention_mask: Context mask [batch_size, context_len]
        
        Returns:
            logits: [batch_size, num_classes]
        """
        batch_size = image.size(0)
        
        # Encode image
        vision_outputs = self.vision_encoder(pixel_values=image)
        vision_features = vision_outputs.last_hidden_state
        vision_features = self.vision_proj(vision_features)
        
        # Encode question
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = text_outputs.last_hidden_state
        text_cls = text_features[:, 0, :]
        
        # Encode retrieved context if provided
        if context_input_ids is not None:
            context_outputs = self.text_encoder(
                input_ids=context_input_ids,
                attention_mask=context_attention_mask
            )
            context_features = context_outputs.last_hidden_state
            context_cls = context_features[:, 0, :]
            
            # Combine question and context
            # Simple approach: average question and context CLS tokens
            text_cls = (text_cls + context_cls) / 2
        
        # Cross-attention: Question attends to image
        attended_vision, _ = self.cross_attention(
            query=text_features,
            key=vision_features,
            value=vision_features,
        )
        attended_vision_cls = attended_vision[:, 0, :]
        
        # Fuse
        combined = torch.cat([text_cls, attended_vision_cls], dim=1)
        fused = self.fusion(combined)
        
        # Classify
        logits = self.classifier(fused)
        
        return logits