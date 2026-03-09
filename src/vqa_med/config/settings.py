"""
Configuration settings for the Medical VQA system.
"""
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


@dataclass
class PathConfig:
    """File path configurations."""
    
    # Data directories
    data_root: Path = PROJECT_ROOT / "data"
    raw_data: Path = data_root / "raw"
    processed_data: Path = data_root / "processed"
    sample_data: Path = data_root / "samples"
    
    # Model directories
    models_dir: Path = PROJECT_ROOT / "models"
    checkpoints_dir: Path = models_dir / "checkpoints"
    
    # Output directories
    outputs_dir: Path = PROJECT_ROOT / "outputs"
    logs_dir: Path = outputs_dir / "logs"
    
    def create_dirs(self):
        """Create all necessary directories."""
        for path in [
            self.raw_data,
            self.processed_data,
            self.sample_data,
            self.checkpoints_dir,
            self.logs_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """Model-related configurations."""
    
    # Base model settings
    vision_model: str = "google/vit-base-patch16-224"
    text_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    
    # Model parameters
    image_size: int = 224
    max_text_length: int = 128
    hidden_dim: int = 768
    num_classes: int = 1000  # Will adjust based on answer vocabulary
    
    # Training parameters
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 10
    device: str = "cuda"  # or "cpu"
    
    # Dropout and regularization
    dropout: float = 0.1


@dataclass
class DataConfig:
    """Data processing configurations."""
    
    # Image preprocessing
    image_mean: tuple = (0.485, 0.456, 0.406)
    image_std: tuple = (0.229, 0.224, 0.225)
    
    # Data splits
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True


class Config:
    """Main configuration class."""
    
    def __init__(self):
        self.paths = PathConfig()
        self.model = ModelConfig()
        self.data = DataConfig()
        
        # Create necessary directories
        self.paths.create_dirs()
    
    def __repr__(self):
        return (
            f"Config(\n"
            f"  paths={self.paths},\n"
            f"  model={self.model},\n"
            f"  data={self.data}\n"
            f")"
        )


# Global config instance
config = Config()