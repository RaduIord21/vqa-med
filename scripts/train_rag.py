"""
Train RAG-enhanced VQA model.
"""
import os
os.environ['MPLBACKEND'] = 'Agg'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
import json
import argparse
import numpy as np

from vqa_med.models import RAGVQAModel
from vqa_med.data import MedicalVQADataset
from vqa_med.utils import (
    get_image_transforms,
    get_tokenizer,
    AverageMeter,
    calculate_accuracy
)
from vqa_med.config import config


class RAGTrainer:
    """Trainer for RAG-VQA model."""
    
    def __init__(
        self, 
        model, 
        train_loader,
        val_loader,
        tokenizer,
        device,
        lr,
        epochs,
        checkpoint_dir,
        use_amp=True,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.num_epochs = epochs
        self.use_amp = use_amp and torch.cuda.is_available()
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer with different learning rates
        pretrained_params = list(self.model.vision_encoder.parameters()) + \
                           list(self.model.text_encoder.parameters())
        new_params = list(self.model.cross_attention.parameters()) + \
                    list(self.model.fusion.parameters()) + \
                    list(self.model.classifier.parameters())
        
        self.optimizer = optim.AdamW([
            {'params': pretrained_params, 'lr': lr * 0.1},
            {'params': new_params, 'lr': lr}
        ], weight_decay=0.01)
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-7
        )
        
        self.scaler = GradScaler() if self.use_amp else None
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
        self.best_val_acc = 0.0
        
        print(f"RAG Trainer initialized on {self.device}")
        if self.use_amp:
            print("✓ Mixed precision enabled")
    
    def process_batch_with_rag(self, batch):
        """
        Process batch and retrieve knowledge for each question.
        
        Returns:
            images, input_ids, attention_mask, labels, context_input_ids, context_attention_mask
        """
        images = batch['image'].to(self.device)
        input_ids = batch['question']['input_ids'].to(self.device)
        attention_mask = batch['question']['attention_mask'].to(self.device)
        labels = batch['answer'].to(self.device)
        
        # Get question texts for retrieval
        questions = batch['question_text']
        
        # Retrieve knowledge (on CPU, then move to GPU)
        contexts = self.model.retrieve_knowledge(questions)
        
        # Tokenize contexts
        context_encoded = self.tokenizer(
            contexts,
            max_length=256,  # Longer for context
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        context_input_ids = context_encoded['input_ids'].to(self.device)
        context_attention_mask = context_encoded['attention_mask'].to(self.device)
        
        return images, input_ids, attention_mask, labels, context_input_ids, context_attention_mask
    
    def train_epoch(self, epoch):
        self.model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs} [Train]")
        
        for batch in pbar:
            # Process batch with RAG
            images, input_ids, attention_mask, labels, context_input_ids, context_attention_mask = \
                self.process_batch_with_rag(batch)
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    logits = self.model(
                        images, input_ids, attention_mask,
                        context_input_ids, context_attention_mask
                    )
                    loss = self.criterion(logits, labels)
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(
                    images, input_ids, attention_mask,
                    context_input_ids, context_attention_mask
                )
                loss = self.criterion(logits, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # Metrics
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
                # Process batch with RAG
                images, input_ids, attention_mask, labels, context_input_ids, context_attention_mask = \
                    self.process_batch_with_rag(batch)
                
                if self.use_amp:
                    with autocast():
                        logits = self.model(
                            images, input_ids, attention_mask,
                            context_input_ids, context_attention_mask
                        )
                        loss = self.criterion(logits, labels)
                else:
                    logits = self.model(
                        images, input_ids, attention_mask,
                        context_input_ids, context_attention_mask
                    )
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
            print(f"✓ Best: {val_acc:.2f}%")
    
    def train(self):
        print("\n" + "=" * 60)
        print("Training RAG-VQA Model")
        print("=" * 60)
        
        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)
            
            self.scheduler.step()
            lr = self.optimizer.param_groups[1]['lr']
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(lr)
            
            print(f"Train: {train_loss:.4f}/{train_acc:.2f}% | Val: {val_loss:.4f}/{val_acc:.2f}% | LR: {lr:.2e}")
            
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            
            self.save_checkpoint(epoch, val_acc, is_best)
        
        print(f"\n✓ Best Validation Accuracy: {self.best_val_acc:.2f}%")
        
        with open(self.checkpoint_dir / "history.json", 'w') as f:
            json.dump(self.history, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_csv', type=str, default=None)
    parser.add_argument('--image_dir', type=str, default=None)
    parser.add_argument('--knowledge_base_path', type=str, required=True,
                        help='Path to medical knowledge base directory')
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--top_k_docs', type=int, default=3,
                        help='Number of documents to retrieve')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 60)
    print("RAG-Enhanced VQA Training")
    print("=" * 60)
    
    # Paths
    data_csv = Path(args.data_csv) if args.data_csv else config.paths.processed_data / "vqa_rad_closed.csv"
    image_dir = Path(args.image_dir) if args.image_dir else config.paths.raw_data / "VQA-RAD" / "images"
    kb_path = Path(args.knowledge_base_path)
    
    if not kb_path.exists():
        print(f"ERROR: Knowledge base not found at {kb_path}")
        print("Please run: uv run python scripts/build_knowledge_base.py --use_sample")
        return
    
    # Data
    transform = get_image_transforms(config.model.image_size, is_training=True)
    tokenizer = get_tokenizer(config.model.text_model)
    
    dataset = MedicalVQADataset(data_csv, image_dir, transform, tokenizer)
    num_classes = dataset.get_num_classes()
    
    # Split
    total = len(dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size
    
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Dataset: Train={train_size}, Val={val_size}, Test={test_size}")
    print(f"Classes: {num_classes}")
    
    # Model
    model = RAGVQAModel(
        num_classes=num_classes,
        knowledge_base_path=str(kb_path),
        top_k_docs=args.top_k_docs,
        dropout=0.2,
    )
    
    # Checkpoint
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else config.paths.checkpoints_dir / "rag"
    
    # Train
    trainer = RAGTrainer(
        model, train_loader, val_loader, tokenizer,
        args.device, args.learning_rate, args.num_epochs, checkpoint_dir
    )
    
    trainer.train()


if __name__ == "__main__":
    main()