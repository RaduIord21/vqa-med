"""
Dataset classes for Medical VQA.
"""
import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional, Tuple
import pandas as pd


class MedicalVQADataset(Dataset):
    """
    Base dataset for Medical Visual Question Answering.
    
    Expected data format:
    - Images in a directory
    - JSON/CSV file with columns: image_path, question, answer
    """
    
    def __init__(
        self,
        data_file: Path,
        image_dir: Path,
        transform=None,
        tokenizer=None,
        max_length: int = 128,
    ):
        """
        Args:
            data_file: Path to CSV/JSON file with questions and answers
            image_dir: Directory containing images
            transform: Image transformation pipeline
            tokenizer: Text tokenizer for questions
            max_length: Maximum length for tokenized questions
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        self.data = self._load_data(data_file)
        
        # Build answer vocabulary
        self.answer_to_idx, self.idx_to_answer = self._build_answer_vocab()
        
    def _load_data(self, data_file: Path) -> pd.DataFrame:
        """Load data from CSV or JSON file."""
        data_file = Path(data_file)
        
        if data_file.suffix == '.csv':
            df = pd.read_csv(data_file)
        elif data_file.suffix == '.json':
            df = pd.read_json(data_file)
        else:
            raise ValueError(f"Unsupported file format: {data_file.suffix}")
        
        # Validate required columns
        required_cols = ['image_path', 'question', 'answer']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Data file must contain columns: {required_cols}")
        
        return df
    
    def _build_answer_vocab(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Build vocabulary mapping for answers."""
        unique_answers = sorted(self.data['answer'].unique())
        answer_to_idx = {ans: idx for idx, ans in enumerate(unique_answers)}
        idx_to_answer = {idx: ans for ans, idx in answer_to_idx.items()}
        return answer_to_idx, idx_to_answer
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary containing:
                - image: Preprocessed image tensor
                - question: Tokenized question
                - answer: Answer label (index)
                - answer_text: Original answer text
        """
        row = self.data.iloc[idx]
        
        # Load and preprocess image
        image_path = self.image_dir / row['image_path']
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Process question
        question = row['question']
        if self.tokenizer:
            question_encoded = self.tokenizer(
                question,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            question_input = {
                'input_ids': question_encoded['input_ids'].squeeze(0),
                'attention_mask': question_encoded['attention_mask'].squeeze(0)
            }
        else:
            question_input = question
        
        # Process answer
        answer_text = row['answer']
        answer_label = self.answer_to_idx[answer_text]
        
        return {
            'image': image,
            'question': question_input,
            'answer': torch.tensor(answer_label, dtype=torch.long),
            'answer_text': answer_text,
            'question_text': question,
        }
    
    def get_num_classes(self) -> int:
        """Return number of unique answers (classes)."""
        return len(self.answer_to_idx)


class MedicalVQADataModule:
    """
    Data module for handling train/val/test splits and dataloaders.
    """
    
    def __init__(
        self,
        data_file: Path,
        image_dir: Path,
        transform,
        tokenizer,
        batch_size: int = 16,
        num_workers: int = 4,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
    ):
        """Initialize data module with split configurations."""
        self.data_file = data_file
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        
        # Will be populated after setup
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def setup(self):
        """Create train/val/test datasets."""
        # Load full dataset
        full_dataset = MedicalVQADataset(
            data_file=self.data_file,
            image_dir=self.image_dir,
            transform=self.transform,
            tokenizer=self.tokenizer,
        )
        
        # Calculate split sizes
        total_size = len(full_dataset)
        train_size = int(total_size * self.train_split)
        val_size = int(total_size * self.val_split)
        test_size = total_size - train_size - val_size
        
        # Split dataset
        from torch.utils.data import random_split
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")
        print(f"Number of answer classes: {full_dataset.get_num_classes()}")
        
        return full_dataset.get_num_classes()
    
    def train_dataloader(self):
        """Return training dataloader."""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        """Return validation dataloader."""
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def test_dataloader(self):
        """Return test dataloader."""
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )