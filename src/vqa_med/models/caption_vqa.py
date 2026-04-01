"""Caption-augmented VQA model with cross-attention fusion."""

import torch
import torch.nn as nn
from transformers import AutoModel, ViTModel


class CaptionVQAModel(nn.Module):
    """
    VQA model that fuses image, question, and image-caption signals.

    Architecture:
    1) ViT image encoder
    2) Shared BioBERT encoder for question and caption
    3) Cross-attention (question -> image patches)
    4) Gated fusion of question, attended vision, and caption
    5) MLP classifier
    """

    def __init__(
        self,
        num_classes: int,
        vision_model_name: str = "google/vit-base-patch16-224",
        text_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        hidden_dim: int = 768,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_classes = num_classes

        print(f"Loading vision model: {vision_model_name}")
        self.vision_encoder = ViTModel.from_pretrained(vision_model_name)
        self.vision_dim = self.vision_encoder.config.hidden_size

        print(f"Loading text model: {text_model_name}")
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.text_dim = self.text_encoder.config.hidden_size

        if self.vision_dim != self.text_dim:
            self.vision_proj = nn.Linear(self.vision_dim, self.text_dim)
        else:
            self.vision_proj = nn.Identity()

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.text_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.feature_gate = nn.Sequential(
            nn.Linear(self.text_dim * 3, self.text_dim * 3),
            nn.LayerNorm(self.text_dim * 3),
            nn.Sigmoid(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(self.text_dim * 3, hidden_dim),
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

    def forward(
        self,
        image: torch.Tensor,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        caption_input_ids: torch.Tensor,
        caption_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for caption-augmented VQA."""
        vision_outputs = self.vision_encoder(pixel_values=image)
        vision_features = self.vision_proj(vision_outputs.last_hidden_state)

        question_outputs = self.text_encoder(
            input_ids=question_input_ids,
            attention_mask=question_attention_mask,
        )
        question_features = question_outputs.last_hidden_state
        question_cls = question_features[:, 0, :]

        caption_outputs = self.text_encoder(
            input_ids=caption_input_ids,
            attention_mask=caption_attention_mask,
        )
        caption_cls = caption_outputs.last_hidden_state[:, 0, :]

        attended_vision, _ = self.cross_attention(
            query=question_features,
            key=vision_features,
            value=vision_features,
        )
        attended_vision_cls = attended_vision[:, 0, :]

        combined = torch.cat([question_cls, attended_vision_cls, caption_cls], dim=1)
        gated = combined * self.feature_gate(combined)
        fused = self.fusion(gated)
        return self.classifier(fused)
