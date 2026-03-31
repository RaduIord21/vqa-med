"""
Utility functions for Medical VQA.
"""
import torch
from torchvision import transforms
from transformers import AutoTokenizer
from typing import Tuple, Optional
import json
import pandas as pd
from pathlib import Path


def get_image_transforms(image_size: int = 224, is_training: bool = True):
    """
    Get image transformation pipeline.
    
    Args:
        image_size: Target image size
        is_training: Whether to apply training augmentations
        
    Returns:
        torchvision transforms composition
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def get_tokenizer(model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"):
    """
    Load pre-trained biomedical tokenizer.
    
    Args:
        model_name: HuggingFace model identifier
        
    Returns:
        Tokenizer instance
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def load_vqa_rad_data(data_dir: Path) -> pd.DataFrame:
    """
    Load and preprocess VQA-RAD dataset.
    
    Expected structure:
    data_dir/
        ├── images/
        │   ├── synpic*.jpg
        └── VQA_RAD Dataset Public.json
    
    Args:
        data_dir: Path to VQA-RAD dataset directory
        
    Returns:
        DataFrame with columns: image_path, question, answer, question_type, answer_type
    """
    data_dir = Path(data_dir)
    json_file = data_dir / "VQA_RAD Dataset Public.json"
    
    if not json_file.exists():
        raise FileNotFoundError(f"VQA-RAD JSON file not found at {json_file}")
    
    # Load JSON data
    with open(json_file, 'r') as f:
        vqa_data = json.load(f)
    
    # Parse data
    records = []
    for item in vqa_data:
        # VQA-RAD format
        image_name = item.get('image_name', '')
        question = item.get('question', '')
        answer = item.get('answer', '')
        question_type = item.get('question_type', 'unknown')
        answer_type = item.get('answer_type', 'unknown')
        
        # Some versions use 'phrase_type' instead of 'question_type'
        if not question_type or question_type == 'unknown':
            question_type = item.get('phrase_type', 'unknown')
        
        # Convert answer to string and normalize (handles both string and int answers)
        answer_str = str(answer).lower().strip() if answer else ''
        
        records.append({
            'image_path': image_name,
            'question': question,
            'answer': answer_str,
            'question_type': question_type,
            'answer_type': answer_type,
        })
    
    df = pd.DataFrame(records)
    
    # Remove any empty entries
    df = df[df['image_path'].notna() & df['question'].notna() & df['answer'].notna()]
    df = df[df['answer'] != '']  # Remove empty string answers
    
    print(f"Loaded {len(df)} QA pairs from VQA-RAD")
    print(f"Unique images: {df['image_path'].nunique()}")
    print(f"Unique answers: {df['answer'].nunique()}")
    print(f"\nQuestion types distribution:")
    print(df['question_type'].value_counts())
    print(f"\nAnswer types distribution:")
    print(df['answer_type'].value_counts())
    
    return df

def prepare_vqa_rad_for_training(
    vqa_rad_dir: Path,
    output_csv: Path,
    filter_answer_type: Optional[str] = None
) -> pd.DataFrame:
    """
    Prepare VQA-RAD data and save as CSV for training.
    
    Args:
        vqa_rad_dir: Directory containing VQA-RAD dataset
        output_csv: Path to save processed CSV
        filter_answer_type: Optional filter (e.g., 'CLOSED' for yes/no, 'OPEN' for free-text)
        
    Returns:
        Processed DataFrame
    """
    df = load_vqa_rad_data(vqa_rad_dir)
    
    # Optional filtering
    if filter_answer_type:
        original_size = len(df)
        df = df[df['answer_type'] == filter_answer_type]
        print(f"\nFiltered to {filter_answer_type} questions: {len(df)}/{original_size}")
    
    # Save to CSV
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    print(f"\nSaved processed data to: {output_csv}")
    
    return df


def get_answer_statistics(df: pd.DataFrame, top_n: int = 20):
    """
    Get statistics about answers in the dataset.
    
    Args:
        df: DataFrame with 'answer' column
        top_n: Number of top answers to display
    """
    print(f"\nTotal unique answers: {df['answer'].nunique()}")
    print(f"\nTop {top_n} most frequent answers:")
    print(df['answer'].value_counts().head(top_n))
    
    # Distribution of answer lengths
    answer_lengths = df['answer'].str.split().str.len()
    print(f"\nAnswer length statistics:")
    print(f"  Mean: {answer_lengths.mean():.2f} words")
    print(f"  Median: {answer_lengths.median():.0f} words")
    print(f"  Max: {answer_lengths.max():.0f} words")


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate classification accuracy.
    
    Args:
        predictions: Predicted class indices [batch_size] or logits [batch_size, num_classes]
        targets: True class indices [batch_size]
        
    Returns:
        Accuracy as percentage
    """
    # Accept either class indices or raw logits.
    if predictions.ndim > 1:
        predictions = torch.argmax(predictions, dim=1)

    if targets.ndim > 1:
        targets = targets.squeeze(-1)

    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    return 100.0 * correct / total