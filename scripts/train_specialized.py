"""
Train specialized models for different question types.
"""
import os
os.environ['MPLBACKEND'] = 'Agg'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import json
import argparse
import numpy as np
import pandas as pd

from vqa_med.models import BaseVQAModel
from vqa_med.data import MedicalVQADataset
from vqa_med.utils import (
    get_image_transforms,
    get_tokenizer,
    AverageMeter,
    calculate_accuracy
)
from vqa_med.config import config


def create_filtered_dataset(data_csv: Path, model_type: str):
    """
    Create filtered dataset for specialized models.
    
    Args:
        data_csv: Original dataset path
        model_type: 'binary' or 'specific' or 'modality'
    """
    df = pd.read_csv(data_csv)
    original_size = len(df)
    
    if model_type == 'binary':
        # Only yes/no questions (PRES, ABN with yes/no answers)
        df = df[df['answer'].isin(['yes', 'no'])]
        print(f"\n[BINARY MODEL] Filtered to yes/no answers: {len(df)}/{original_size}")
        
    elif model_type == 'specific':
        # Exclude yes/no questions
        df = df[~df['answer'].isin(['yes', 'no'])]
        print(f"\n[SPECIFIC MODEL] Excluded yes/no: {len(df)}/{original_size}")
        
    elif model_type == 'modality':
        # Only MODALITY and PLANE questions
        df = df[df['question_type'].isin(['MODALITY', 'PLANE'])]
        print(f"\n[MODALITY MODEL] Filtered to MODALITY/PLANE: {len(df)}/{original_size}")
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    if len(df) == 0:
        raise ValueError("No samples after filtering!")
    
    # Save filtered dataset
    filtered_path = data_csv.parent / f"vqa_rad_{model_type}.csv"
    df.to_csv(filtered_path, index=False)
    
    print(f"Dataset saved to: {filtered_path}")
    print(f"\nAnswer distribution:")
    print(df['answer'].value_counts().head(20))
    print(f"\nQuestion type distribution:")
    print(df['question_type'].value_counts().head(10))
    
    return filtered_path


class SimpleTrainer:
    """Simplified trainer for specialized models."""
    
    def __init__(self, model, train_loader, val_loader, device, lr, epochs, checkpoint_dir):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = epochs
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-7
        )
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
        self.best_val_acc = 0.0
    
    def train_epoch(self, epoch):
        self.model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs} [Train]")
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            input_ids = batch['question']['input_ids'].to(self.device)
            attention_mask = batch['question']['attention_mask'].to(self.device)
            labels = batch['answer'].to(self.device)
            
            logits = self.model(images, input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            predictions = torch.argmax(logits, dim=1)
            acc = calculate_accuracy(predictions, labels)
            
            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(acc, images.size(0))
            
            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}', 'acc': f'{acc_meter.avg:.2f}%'})
        
        return loss_meter.avg, acc_meter.avg
    
    def validate(self, epoch):
        self.model.eval()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        
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
                
                loss_meter.update(loss.item(), images.size(0))
                acc_meter.update(acc, images.size(0))
                
                pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}', 'acc': f'{acc_meter.avg:.2f}%'})
        
        return loss_meter.avg, acc_meter.avg
    
    def save_checkpoint(self, epoch, val_acc, is_best):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'history': self.history,
        }
        
        torch.save(checkpoint, self.checkpoint_dir / "checkpoint_latest.pth")
        
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "checkpoint_best.pth")
            print(f"✓ Best model: {val_acc:.2f}%")
    
    def train(self):
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)
        
        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)
            
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(lr)
            
            print(f"Train: {train_loss:.4f} / {train_acc:.2f}% | Val: {val_loss:.4f} / {val_acc:.2f}% | LR: {lr:.2e}")
            
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            
            self.save_checkpoint(epoch, val_acc, is_best)
        
        print(f"\nBest Validation Accuracy: {self.best_val_acc:.2f}%")
        
        with open(self.checkpoint_dir / "history.json", 'w') as f:
            json.dump(self.history, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['binary', 'specific', 'modality'],
                        help='Type of specialized model to train')
    parser.add_argument('--data_csv', type=str, default=None)
    parser.add_argument('--image_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 60)
    print(f"Training Specialized Model: {args.model_type.upper()}")
    print("=" * 60)
    
    # Paths
    data_csv = Path(args.data_csv) if args.data_csv else config.paths.processed_data / "vqa_rad_closed.csv"
    image_dir = Path(args.image_dir) if args.image_dir else config.paths.raw_data / "VQA-RAD" / "images"
    
    # Create filtered dataset
    filtered_csv = create_filtered_dataset(data_csv, args.model_type)
    
    # Load data
    transform = get_image_transforms(config.model.image_size, is_training=True)
    tokenizer = get_tokenizer(config.model.text_model)
    
    full_dataset = MedicalVQADataset(
        data_file=filtered_csv,
        image_dir=image_dir,
        transform=transform,
        tokenizer=tokenizer,
    )
    
    num_classes = full_dataset.get_num_classes()
    print(f"\nNumber of classes: {num_classes}")
    
    # Split
    from torch.utils.data import random_split
    total = len(full_dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size
    
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"Split: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Model
    model = BaseVQAModel(
        num_classes=num_classes,
        vision_model_name=config.model.vision_model,
        text_model_name=config.model.text_model,
        hidden_dim=config.model.hidden_dim,
        dropout=0.3,
    )
    
    # Checkpoint dir
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
    else:
        checkpoint_dir = config.paths.checkpoints_dir / f"specialized_{args.model_type}"
    
    # Train
    trainer = SimpleTrainer(
        model, train_loader, val_loader,
        args.device, args.learning_rate, args.num_epochs, checkpoint_dir
    )
    
    trainer.train()


if __name__ == "__main__":
    main()