"""
Attention-based VQA model with better visual-textual fusion.
"""
import torch
import torch.nn as nn
from transformers import ViTModel, AutoModel


class AttentionVQAModel(nn.Module):
    """
    VQA model with cross-attention between vision and text.
    
    Architecture:
    1. Vision Encoder: ViT
    2. Text Encoder: BioBERT
    3. Cross-Attention: Text attends to visual features
    4. Fusion: Attended features + text features
    5. Classifier: Answer prediction
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
        self.hidden_dim = hidden_dim
        
        # Vision Encoder
        print(f"Loading vision model: {vision_model_name}")
        self.vision_encoder = ViTModel.from_pretrained(vision_model_name)
        self.vision_dim = self.vision_encoder.config.hidden_size
        
        # Text Encoder
        print(f"Loading text model: {text_model_name}")
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.text_dim = self.text_encoder.config.hidden_size
        
        # Project to same dimension if needed
        if self.vision_dim != self.text_dim:
            self.vision_proj = nn.Linear(self.vision_dim, self.text_dim)
        else:
            self.vision_proj = nn.Identity()
        
        # Cross-Attention: Text queries attend to image patches
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.text_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(self.text_dim * 2, hidden_dim),  # text + attended_vision
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
        
        print(f"Attention VQA Model initialized:")
        print(f"  Vision dim: {self.vision_dim}")
        print(f"  Text dim: {self.text_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Attention heads: {num_attention_heads}")
        print(f"  Num classes: {num_classes}")
    
    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with cross-attention.
        
        Args:
            image: [batch_size, 3, H, W]
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            logits: [batch_size, num_classes]
        """
        batch_size = image.size(0)
        
        # Encode image - get all patch embeddings
        vision_outputs = self.vision_encoder(pixel_values=image)
        vision_features = vision_outputs.last_hidden_state  # [B, num_patches, vision_dim]
        vision_features = self.vision_proj(vision_features)  # [B, num_patches, text_dim]
        
        # Encode question - get all token embeddings
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = text_outputs.last_hidden_state  # [B, seq_len, text_dim]
        text_cls = text_features[:, 0, :]  # [B, text_dim] - CLS token
        
        # Cross-Attention: Question attends to image patches
        # Query: text features, Key/Value: vision features
        attended_vision, attention_weights = self.cross_attention(
            query=text_features,  # [B, seq_len, text_dim]
            key=vision_features,   # [B, num_patches, text_dim]
            value=vision_features, # [B, num_patches, text_dim]
        )
        # attended_vision: [B, seq_len, text_dim]
        
        # Use CLS token's attended vision
        attended_vision_cls = attended_vision[:, 0, :]  # [B, text_dim]
        
        # Fuse text and attended vision
        combined = torch.cat([text_cls, attended_vision_cls], dim=1)  # [B, text_dim*2]
        fused = self.fusion(combined)  # [B, hidden_dim]
        
        # Classify
        logits = self.classifier(fused)  # [B, num_classes]
        
        return logits
    
    def get_attention_maps(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        """Get attention maps for visualization."""
        with torch.no_grad():
            vision_outputs = self.vision_encoder(pixel_values=image)
            vision_features = vision_outputs.last_hidden_state
            vision_features = self.vision_proj(vision_features)
            
            text_outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            text_features = text_outputs.last_hidden_state
            
            _, attention_weights = self.cross_attention(
                query=text_features,
                key=vision_features,
                value=vision_features,
            )
            
        return attention_weights


class AttentionVQAModelWrapper:
    """Wrapper for AttentionVQAModel."""
    
    def __init__(self, model: AttentionVQAModel, device: str = "cuda"):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Model moved to: {self.device}")
    
    def predict(self, image, input_ids, attention_mask):
        """Make predictions."""
        self.model.eval()
        
        with torch.no_grad():
            image = image.to(self.device)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            
            logits = self.model(image, input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=-1)
            predicted_indices = torch.argmax(logits, dim=-1)
        
        return predicted_indices, probabilities
    
    def save_checkpoint(self, filepath, epoch, optimizer_state=None):
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.model.num_classes,
        }
        if optimizer_state:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath):
        """Load checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Checkpoint loaded: {filepath}")
        return checkpoint