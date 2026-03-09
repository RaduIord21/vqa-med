"""
Base VQA Model Architecture.
Combines vision encoder (ViT) and text encoder (BERT) for medical VQA.
"""
import torch
import torch.nn as nn
from transformers import ViTModel, AutoModel
from typing import Dict, Tuple


class BaseVQAModel(nn.Module):
    """
    Base Visual Question Answering model.
    
    Architecture:
    1. Vision Encoder: ViT for image feature extraction
    2. Text Encoder: BioBERT for question encoding
    3. Fusion: Concatenation + MLP
    4. Classifier: Linear layer for answer prediction
    """
    
    def __init__(
        self,
        num_classes: int,
        vision_model_name: str = "google/vit-base-patch16-224",
        text_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        hidden_dim: int = 768,
        dropout: float = 0.1,
        freeze_vision: bool = False,
        freeze_text: bool = False,
    ):
        """
        Args:
            num_classes: Number of answer classes
            vision_model_name: HuggingFace vision model identifier
            text_model_name: HuggingFace text model identifier
            hidden_dim: Hidden dimension size
            dropout: Dropout probability
            freeze_vision: Whether to freeze vision encoder
            freeze_text: Whether to freeze text encoder
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Vision Encoder (ViT)
        print(f"Loading vision model: {vision_model_name}")
        self.vision_encoder = ViTModel.from_pretrained(vision_model_name)
        self.vision_dim = self.vision_encoder.config.hidden_size
        
        if freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            print("Vision encoder frozen")
        
        # Text Encoder (BioBERT)
        print(f"Loading text model: {text_model_name}")
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.text_dim = self.text_encoder.config.hidden_size
        
        if freeze_text:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            print("Text encoder frozen")
        
        # Fusion Layer
        self.fusion = nn.Sequential(
            nn.Linear(self.vision_dim + self.text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        print(f"Model initialized:")
        print(f"  Vision dim: {self.vision_dim}")
        print(f"  Text dim: {self.text_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Num classes: {num_classes}")
    
    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            image: Image tensor [batch_size, 3, H, W]
            input_ids: Tokenized question [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            logits: Class logits [batch_size, num_classes]
        """
        # Encode image
        vision_outputs = self.vision_encoder(pixel_values=image)
        # Use [CLS] token representation
        image_features = vision_outputs.last_hidden_state[:, 0, :]  # [batch_size, vision_dim]
        
        # Encode question
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Use [CLS] token representation
        text_features = text_outputs.last_hidden_state[:, 0, :]  # [batch_size, text_dim]
        
        # Fuse features
        combined_features = torch.cat([image_features, text_features], dim=1)  # [batch_size, vision_dim + text_dim]
        fused_features = self.fusion(combined_features)  # [batch_size, hidden_dim]
        
        # Classify
        logits = self.classifier(fused_features)  # [batch_size, num_classes]
        
        return logits
    
    def get_embeddings(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get intermediate embeddings (useful for analysis).
        
        Returns:
            image_features, text_features, fused_features
        """
        with torch.no_grad():
            # Encode image
            vision_outputs = self.vision_encoder(pixel_values=image)
            image_features = vision_outputs.last_hidden_state[:, 0, :]
            
            # Encode question
            text_outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            text_features = text_outputs.last_hidden_state[:, 0, :]
            
            # Fuse
            combined_features = torch.cat([image_features, text_features], dim=1)
            fused_features = self.fusion(combined_features)
        
        return image_features, text_features, fused_features


class VQAModelWrapper:
    """
    Wrapper class for easier model management.
    Handles device placement, inference, and utilities.
    """
    
    def __init__(self, model: BaseVQAModel, device: str = "cuda"):
        """
        Args:
            model: Base VQA model instance
            device: Device to run model on
        """
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Model moved to: {self.device}")
    
    def predict(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions.
        
        Returns:
            predicted_indices, probabilities
        """
        self.model.eval()
        
        with torch.no_grad():
            # Move to device
            image = image.to(self.device)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            
            # Forward pass
            logits = self.model(image, input_ids, attention_mask)
            
            # Get predictions
            probabilities = torch.softmax(logits, dim=-1)
            predicted_indices = torch.argmax(logits, dim=-1)
        
        return predicted_indices, probabilities
    
    def save_checkpoint(self, filepath: str, epoch: int, optimizer_state: dict = None):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.model.num_classes,
        }
        
        if optimizer_state:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Checkpoint loaded from: {filepath}")
        return checkpoint
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        vision_params = sum(p.numel() for p in self.model.vision_encoder.parameters())
        text_params = sum(p.numel() for p in self.model.text_encoder.parameters())
        fusion_params = sum(p.numel() for p in self.model.fusion.parameters())
        classifier_params = sum(p.numel() for p in self.model.classifier.parameters())
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'vision': vision_params,
            'text': text_params,
            'fusion': fusion_params,
            'classifier': classifier_params,
        }