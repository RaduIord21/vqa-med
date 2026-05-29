"""RAG + caption-augmented VQA model.

This model combines:
- Vision encoder (ViT)
- Text encoder (PubMedBERT/BioBERT) shared across question, caption, and retrieved context
- Retrieval-Augmented context encoding (RAG)
- Cross-attention fusion (question -> context, question -> vision)
- Late fusion of (question+context), attended vision, and caption

Adversarial prompting is intentionally kept outside the model as a preprocessing
step (question string rewriting) to avoid baking non-differentiable text rules
into the network.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, ViTModel

from ..retrieval import MedicalKnowledgeBase, MedicalRetriever


class RAGCaptionVQAModel(nn.Module):
    """Combined RAG + caption VQA model."""

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
        use_gated_qc_fusion: bool = True,
        use_feature_gate: bool = True,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.top_k_docs = top_k_docs
        self.use_gated_qc_fusion = use_gated_qc_fusion
        self.use_feature_gate = use_feature_gate

        # Retrieval
        self.knowledge_base = MedicalKnowledgeBase.load(knowledge_base_path)
        self.retriever = MedicalRetriever(self.knowledge_base, top_k=top_k_docs)

        # Encoders
        self.vision_encoder = ViTModel.from_pretrained(vision_model_name)
        self.vision_dim = self.vision_encoder.config.hidden_size

        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.text_dim = self.text_encoder.config.hidden_size

        # Project vision to text dimension
        if self.vision_dim != self.text_dim:
            self.vision_proj = nn.Linear(self.vision_dim, self.text_dim)
        else:
            self.vision_proj = nn.Identity()

        # Cross-attention: question tokens attend to retrieved context
        self.context_attention = nn.MultiheadAttention(
            embed_dim=self.text_dim,
            num_heads=max(1, num_attention_heads // 2),
            dropout=dropout,
            batch_first=True,
        )

        # Cross-attention: (question + attended context) attends to vision patches
        self.vision_attention = nn.MultiheadAttention(
            embed_dim=self.text_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Gate question vs context at CLS level
        if use_gated_qc_fusion:
            self.qc_gate = nn.Sequential(
                nn.Linear(self.text_dim * 2, self.text_dim),
                nn.Sigmoid(),
            )
        else:
            self.qc_gate = None

        # Late fusion: (question+context) CLS + attended vision CLS + caption CLS
        fused_in_dim = self.text_dim * 3
        if use_feature_gate:
            self.feature_gate = nn.Sequential(
                nn.Linear(fused_in_dim, fused_in_dim),
                nn.LayerNorm(fused_in_dim),
                nn.Sigmoid(),
            )
        else:
            self.feature_gate = None

        self.fusion = nn.Sequential(
            nn.Linear(fused_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def retrieve_knowledge(
        self,
        questions: List[str],
        image_captions: Optional[List[str]] = None,
        image_features: Optional[torch.Tensor] = None,
        visual_weight: Optional[float] = None,
    ) -> List[str]:
        """Retrieve formatted context strings for a batch."""
        contexts: List[str] = []
        if image_captions is None:
            image_captions = [None for _ in questions]

        for idx, question in enumerate(questions):
            caption = image_captions[idx] if idx < len(image_captions) else None
            img_feat = None
            if image_features is not None:
                img_feat = image_features[idx : idx + 1]

            retrieved_docs = self.retriever.retrieve(
                question,
                image_caption=caption,
                image_features=img_feat,
                visual_weight=visual_weight,
            )
            context = self.retriever.format_context(retrieved_docs)
            contexts.append(context if context else "")
        return contexts

    def forward(
        self,
        image: torch.Tensor,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        context_input_ids: Optional[torch.Tensor] = None,
        context_attention_mask: Optional[torch.Tensor] = None,
        caption_input_ids: Optional[torch.Tensor] = None,
        caption_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            image: [B, 3, H, W]
            question_input_ids/mask: [B, q_len]
            context_input_ids/mask: [B, c_len] (optional)
            caption_input_ids/mask: [B, cap_len] (required for caption fusion)

        Returns:
            logits: [B, num_classes]
        """
        # Encode vision tokens
        vision_outputs = self.vision_encoder(pixel_values=image)
        vision_features = self.vision_proj(vision_outputs.last_hidden_state)  # [B, P, D]

        # Encode question tokens
        q_outputs = self.text_encoder(
            input_ids=question_input_ids,
            attention_mask=question_attention_mask,
        )
        q_features = q_outputs.last_hidden_state  # [B, Q, D]
        q_cls = q_features[:, 0, :]

        # Encode caption tokens
        if caption_input_ids is None or caption_attention_mask is None:
            raise ValueError("caption_input_ids and caption_attention_mask are required")
        cap_outputs = self.text_encoder(
            input_ids=caption_input_ids,
            attention_mask=caption_attention_mask,
        )
        cap_cls = cap_outputs.last_hidden_state[:, 0, :]

        # Encode / attend over retrieved context
        attended_context = torch.zeros_like(q_features)
        context_cls = torch.zeros_like(q_cls)
        if context_input_ids is not None and context_attention_mask is not None and context_input_ids.size(1) > 0:
            ctx_outputs = self.text_encoder(
                input_ids=context_input_ids,
                attention_mask=context_attention_mask,
            )
            ctx_features = ctx_outputs.last_hidden_state  # [B, C, D]

            attended_context, _ = self.context_attention(
                query=q_features,
                key=ctx_features,
                value=ctx_features,
            )
            context_cls = attended_context[:, 0, :]

        # Fuse question + context at CLS level
        if self.qc_gate is not None:
            gate = self.qc_gate(torch.cat([q_cls, context_cls], dim=1))
            text_with_context = gate * q_cls + (1.0 - gate) * context_cls
        else:
            text_with_context = (q_cls + context_cls) / 2

        # Enhance token-level query for vision attention
        enhanced_q_features = q_features + attended_context

        # Cross-attention: enhanced text attends to vision
        attended_vision, _ = self.vision_attention(
            query=enhanced_q_features,
            key=vision_features,
            value=vision_features,
        )
        vision_cls = attended_vision[:, 0, :]

        # Late fusion
        combined = torch.cat([text_with_context, vision_cls, cap_cls], dim=1)
        if self.feature_gate is not None:
            combined = combined * self.feature_gate(combined)

        fused = self.fusion(combined)
        logits = self.classifier(fused)
        return logits
