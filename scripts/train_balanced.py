"""
Improved training script with class balancing and question-type awareness.
"""
import os
os.environ['MPLBACKEND'] = 'Agg'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import argparse
import numpy as np
import pandas as pd

from vqa_med.models import BaseVQAModel, VQAModelWrapper
from vqa_med.data import MedicalVQADataset
from vqa_med.utils import (
    get_image_transforms,
    get_tokenizer,
    AverageMeter,
    calculate_accuracy
)
from vqa_med.config import config


def get_balanced_sampler(dataset, balance_strategy='answer'):
    """
    Create weighted sampler for balanced training.
    
    Args:
        dataset: VQA dataset
        balance_strategy: 'answer' or 'question_type'
    """
    if balance_strategy == 'answer':
        # Balance by answer frequency
        all_answers = [dataset.data.iloc[i]['answer'] for i in range(len(dataset))]
        answer_indices = [dataset.answer_to_idx[ans] for ans in all_answers]
        
        # Calculate class weights (inverse frequency)
        unique, counts = np.unique(answer_indices, return_counts=True)
        class_weights = 1.0 / counts
        
        # Assign weight to each sample based on its answer
        sample_weights = np.array([class_weights[np.where(unique == idx)[0][0]] for idx in answer_indices])
        
    elif balance_strategy == 'question_type':
        # Balance by question type
        all_qt = [dataset.data.iloc[i]['question_type'] for i in range(len(dataset))]
        
        # Calculate question type weights
        from collections import Counter
        qt_counts = Counter(all_qt)
        qt_weights = {qt: 1.0 / count for qt, count in qt_counts.items()}
        
        # Assign weight to each sample
        sample_weights = np.array([qt_weights[qt] for qt in all_qt])
    
    else:
        raise ValueError(f"Unknown balance_strategy: {balance_strategy}")
    
    # Normalize weights
    sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)
    
    print(f"\nCreated balanced sampler with strategy: {balance_strategy}")
    print(f"Weight range: {sample_weights.min():.4f} - {sample_weights.max():.4f}")
    
    return WeightedRandomSampler(
        weights=torch.FloatTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )


def filter_dataset_by_answer_type(data_csv: Path, keep_yes_no_only=False, exclude_yes_no=False):
    """Filter dataset to focus on specific answer types."""
    df = pd.read_csv(data_csv)
    
    original_size = len(df)
    
    if keep_yes_no_only:
        df = df[df['answer'].isin(['yes', 'no'])]
        print(f"\nFiltered to yes/no only: {len(df)}/{original_size} samples")
    elif exclude_yes_no:
        df = df[~df['answer'].isin(['yes', 'no'])]
        print(f"\nExcluded yes/no: {len(df)}/{original_size} samples")
    
    if len(df) == 0:
        raise ValueError("No samples left after filtering!")
    
    # Save filtered dataset
    suffix = "yesno" if keep_yes_no_only else "specific" if exclude_yes_no else "all"
    filtered_path = data_csv.parent / f"vqa_rad_closed_{suffix}.csv"
    df.to_csv(filtered_path, index=False)
    
    print(f"Filtered dataset saved to: {filtered_path}")
    print(f"Answer distribution:")
    print(df['answer'].value_counts().head(20))
    
    return filtered_path


class ImprovedVQATrainer:
    """Improved trainer with better strategies."""
    
    def __init__(
        self,
        model: BaseVQAModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
        learning_rate: float = 1e-4,
        num_epochs: int = 20,
        checkpoint_dir: Path = None,
        use_focal_loss: bool = False,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        
        # Loss function
        if use_focal_loss:
            print("Using Focal Loss (for class imbalance)")
            self.criterion = FocalLoss(alpha=1.0, gamma=2.0)
        else:
            print("Using CrossEntropy Loss")
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler - more aggressive
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        )
        
        # Warmup scheduler
        self.warmup_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=5
        )
        
        self.checkpoint_dir = checkpoint_dir or config.paths.checkpoints_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        self.best_val_acc = 0.0
        self.current_epoch = 0
        
        print(f"Trainer initialized on device: {self.device}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
    
    def train_epoch(self, epoch: int) -> tuple:
        """Train for one epoch."""
        self.model.train()
        
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            input_ids = batch['question']['input_ids'].to(self.device)
            attention_mask = batch['question']['attention_mask'].to(self.device)
            labels = batch['answer'].to(self.device)
            
            # Forward pass
            logits = self.model(images, input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Metrics
            predictions = torch.argmax(logits, dim=1)
            acc = calculate_accuracy(predictions, labels)
            
            batch_size = images.size(0)
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(acc, batch_size)
            
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'acc': f'{acc_meter.avg:.2f}%'
            })
        
        # Warmup scheduler for first 5 epochs
        if epoch <= 5:
            self.warmup_scheduler.step()
        
        return loss_meter.avg, acc_meter.avg
    
    def validate(self, epoch: int) -> tuple:
        """Validate the model."""
        self.model.eval()
        
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.num_epochs} [Val]")
        
        with torch.no_grad():
            for batch in pbar:
                images = batch['image'].to(self.device)
                input_ids = batch['question']['input_ids'].to(self.device)
                attention_mask = batch['question']['attention_mask'].to(self.device)
                labels = batch['answer'].to(self.device)
                
                logits = self.model(images, input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                
                predictions = torch.argmax(logits, dim=1)
                acc = calculate_accuracy(predictions, labels)
                
                batch_size = images.size(0)
                loss_meter.update(loss.item(), batch_size)
                acc_meter.update(acc, batch_size)
                
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({
                    'loss': f'{loss_meter.avg:.4f}',
                    'acc': f'{acc_meter.avg:.2f}%'
                })
        
        return loss_meter.avg, acc_meter.avg
    
    def save_checkpoint(self, epoch: int, val_acc: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'best_val_acc': self.best_val_acc,
            'history': self.history,
        }
        
        latest_path = self.checkpoint_dir / "checkpoint_latest.pth"
        torch.save(checkpoint, latest_path)
        
        if is_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pth"
            torch.save(checkpoint, best_path)
            print(f"✓ New best model saved (acc: {val_acc:.2f}%)")
        
        # Save periodic checkpoints
        if epoch % 5 == 0:
            epoch_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(checkpoint, epoch_path)
    
    def train(self, resume_from: Path = None):
        """Full training loop."""
        start_epoch = 1
        
        if resume_from:
            checkpoint = torch.load(resume_from, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.history = checkpoint['history']
            self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resumed from epoch {checkpoint['epoch']}")
        
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)
        
        for epoch in range(start_epoch, self.num_epochs + 1):
            self.current_epoch = epoch
            
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            print("-" * 60)
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            
            # Update learning rate (after warmup)
            if epoch > 5:
                old_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step(val_acc)
                new_lr = self.optimizer.param_groups[0]['lr']
                if new_lr != old_lr:
                    print(f"Learning rate reduced: {old_lr:.2e} → {new_lr:.2e}")
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # Print summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"  Learning Rate: {current_lr:.2e}")
            
            # Save checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            
            self.save_checkpoint(epoch, val_acc, is_best)
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Best Validation Accuracy: {self.best_val_acc:.2f}%")
        
        # Save history
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to: {history_path}")


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train VQA Model with Balancing')
    
    # Data arguments
    parser.add_argument('--data_csv', type=str, default=None)
    parser.add_argument('--image_dir', type=str, default=None)
    parser.add_argument('--data_type', type=str, default='full',
                        choices=['closed', 'full'],
                        help='Use closed-ended or full dataset (default: full for all answer types)')
    parser.add_argument('--filter_data', type=str, default='none',
                        choices=['none', 'yesno_only', 'exclude_yesno'],
                        help='Filter dataset by answer type')
    
    # Training strategies
    parser.add_argument('--balance_strategy', type=str, default='answer',
                        choices=['none', 'answer', 'question_type'],
                        help='Sampling balance strategy')
    parser.add_argument('--use_focal_loss', action='store_true',
                        help='Use focal loss instead of cross entropy')
    
    # Model arguments
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Other arguments
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 60)
    print("VQA Training - Balanced & Improved")
    print("=" * 60)
    
    # Paths
    if args.data_csv:
        data_csv = Path(args.data_csv)
    else:
        data_csv = config.paths.processed_data / f"vqa_rad_{args.data_type}.csv"
    
    if args.image_dir:
        image_dir = Path(args.image_dir)
    else:
        image_dir = config.paths.raw_data / "VQA-RAD" / "images"
    
    # Filter data if requested
    if args.filter_data == 'yesno_only':
        data_csv = filter_dataset_by_answer_type(data_csv, keep_yes_no_only=True)
    elif args.filter_data == 'exclude_yesno':
        data_csv = filter_dataset_by_answer_type(data_csv, exclude_yes_no=True)
    
    # Check data
    if not data_csv.exists():
        print(f"ERROR: Data file not found at {data_csv}")
        return
    
    print(f"\nConfiguration:")
    print(f"  Data: {data_csv.name}")
    print(f"  Balance Strategy: {args.balance_strategy}")
    print(f"  Use Focal Loss: {args.use_focal_loss}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Epochs: {args.num_epochs}")
    
    # Prepare data
    print("\nPreparing data...")
    
    train_transform = get_image_transforms(config.model.image_size, is_training=True)
    val_transform = get_image_transforms(config.model.image_size, is_training=False)
    tokenizer = get_tokenizer(config.model.text_model)
    
    # Load full dataset
    full_dataset = MedicalVQADataset(
        data_file=data_csv,
        image_dir=image_dir,
        transform=train_transform,
        tokenizer=tokenizer,
    )
    
    num_classes = full_dataset.get_num_classes()
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(total_size * 0.7)
    val_size = int(total_size * 0.15)
    test_size = total_size - train_size - val_size
    
    from torch.utils.data import random_split
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"Split: Train={train_size}, Val={val_size}, Test={test_size}")
    print(f"Number of answer classes: {num_classes}")
    
    # Create dataloaders
    if args.balance_strategy != 'none':
        # Create temporary dataset for train split to get sampler
        train_indices = train_dataset.indices
        
        # Create a view of the data for these indices
        class SubsetDataset:
            def __init__(self, dataset, indices):
                self.dataset = dataset
                self.indices = indices
                self.data = dataset.data.iloc[indices].reset_index(drop=True)
                self.answer_to_idx = dataset.answer_to_idx
            
            def __len__(self):
                return len(self.indices)
        
        temp_train = SubsetDataset(full_dataset, train_indices)
        sampler = get_balanced_sampler(temp_train, args.balance_strategy)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Create model
    print("\nInitializing model...")
    model = BaseVQAModel(
        num_classes=num_classes,
        vision_model_name=config.model.vision_model,
        text_model_name=config.model.text_model,
        hidden_dim=config.model.hidden_dim,
        dropout=0.2,  # Slightly higher dropout
    )
    
    # Checkpoint directory
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = config.paths.checkpoints_dir / f"balanced_{timestamp}"
    
    # Create trainer
    trainer = ImprovedVQATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        checkpoint_dir=checkpoint_dir,
        use_focal_loss=args.use_focal_loss,
    )
    
    # Train
    resume_from = Path(args.resume_from) if args.resume_from else None
    trainer.train(resume_from=resume_from)


if __name__ == "__main__":
    main()