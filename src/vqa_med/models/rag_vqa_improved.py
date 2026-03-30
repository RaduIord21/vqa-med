"""
Improved RAG-VQA with better fusion and retrieval strategies.
"""
import torch
import torch.nn as nn
from transformers import ViTModel, AutoModel
from typing import Optional, List

from ..retrieval import MedicalKnowledgeBase, MedicalRetriever


class ImprovedRAGVQAModel(nn.Module):
    """
    Improved RAG-VQA with:
    1. Cross-attention fusion between question and retrieved context
    2. Gated fusion mechanism
    3. Better integration of visual, textual, and retrieved information
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
        use_gated_fusion: bool = True,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.top_k_docs = top_k_docs
        self.use_gated_fusion = use_gated_fusion
        
        # Load knowledge base
        print(f"Loading knowledge base from: {knowledge_base_path}")
        self.knowledge_base = MedicalKnowledgeBase.load(knowledge_base_path)
        self.retriever = MedicalRetriever(self.knowledge_base, top_k=top_k_docs)
        
        # Vision Encoder
        print(f"Loading vision model: {vision_model_name}")
        self.vision_encoder = ViTModel.from_pretrained(vision_model_name)
        self.vision_dim = self.vision_encoder.config.hidden_size
        
        # Text Encoder
        print(f"Loading text model: {text_model_name}")
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.text_dim = self.text_encoder.config.hidden_size
        
        # Project vision to text dimension
        if self.vision_dim != self.text_dim:
            self.vision_proj = nn.Linear(self.vision_dim, self.text_dim)
        else:
            self.vision_proj = nn.Identity()
        
        # Cross-attention between question and retrieved context
        self.context_attention = nn.MultiheadAttention(
            embed_dim=self.text_dim,
            num_heads=num_attention_heads // 2,  # Fewer heads for context
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-attention between text and vision
        self.vision_attention = nn.MultiheadAttention(
            embed_dim=self.text_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Gated fusion for combining question + context
        if use_gated_fusion:
            self.fusion_gate = nn.Sequential(
                nn.Linear(self.text_dim * 2, self.text_dim),
                nn.Sigmoid()
            )
        
        # Multi-modal fusion
        self.multimodal_fusion = nn.Sequential(
            nn.Linear(self.text_dim * 3, hidden_dim),  # question + context + vision
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
        
        print(f"Improved RAG-VQA Model initialized:")
        print(f"  Gated fusion: {use_gated_fusion}")
        print(f"  Top-K docs: {top_k_docs}")
    
    def retrieve_knowledge(self, questions: List[str]) -> List[str]:
        """Retrieve relevant medical knowledge."""
        contexts = []
        for question in questions:
            retrieved_docs = self.retriever.retrieve(question)
            context = self.retriever.format_context(retrieved_docs)
            contexts.append(context if context else "No relevant information found.")
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
        Forward pass with improved RAG fusion.
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
        question_cls = text_features[:, 0, :]
        
        # Encode retrieved context
        if context_input_ids is not None and context_input_ids.size(1) > 0:
            context_outputs = self.text_encoder(
                input_ids=context_input_ids,
                attention_mask=context_attention_mask
            )
            context_features = context_outputs.last_hidden_state
            
            # Cross-attention: Question attends to context
            attended_context, _ = self.context_attention(
                query=text_features,
                key=context_features,
                value=context_features,
            )
            context_cls = attended_context[:, 0, :]
            
            # Gated fusion of question and context
            if self.use_gated_fusion:
                gate = self.fusion_gate(torch.cat([question_cls, context_cls], dim=1))
                text_with_context = gate * question_cls + (1 - gate) * context_cls
            else:
                text_with_context = (question_cls + context_cls) / 2
        else:
            text_with_context = question_cls
            context_cls = torch.zeros_like(question_cls)
        
        # Cross-attention: Enhanced text attends to vision
        attended_vision, _ = self.vision_attention(
            query=text_features,
            key=vision_features,
            value=vision_features,
        )
        vision_cls = attended_vision[:, 0, :]
        
        # Combine all three modalities
        combined = torch.cat([question_cls, context_cls, vision_cls], dim=1)
        fused = self.multimodal_fusion(combined)
        
        # Classify
        logits = self.classifier(fused)
        
        return logits