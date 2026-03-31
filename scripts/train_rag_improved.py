"""
Train Improved RAG-enhanced VQA model with visual-aware retrieval.

Better fusion strategies:
- Gated fusion between question and context
- Cross-attention between text and vision
- Three-way multimodal fusion (question + context + vision)
- Visual-aware document retrieval
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

from vqa_med.models import ImprovedRAGVQAModel
from vqa_med.data import MedicalVQADataset
from vqa_med.utils import (
    get_image_transforms,
    get_tokenizer,
    AverageMeter,
    calculate_accuracy
)
from vqa_med.config import config


class ImprovedRAGTrainer:
    """Trainer for Improved RAG-VQA model with better fusion and visual-aware retrieval."""
    
    def __init__(
        self, 
        model, 
        train_loader,
        val_loader,
        tokenizer,
        device,
        lr,
        epochs,
        gradient_accumulation_steps,
        checkpoint_dir,
        use_amp=True,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.num_epochs = epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_amp = use_amp and torch.cuda.is_available()
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer with different learning rates for pretrained vs new layers
        pretrained_params = list(self.model.vision_encoder.parameters()) + \
                           list(self.model.text_encoder.parameters())
        new_params = list(self.model.context_attention.parameters()) + \
                    list(self.model.vision_attention.parameters()) + \
                    list(self.model.multimodal_fusion.parameters()) + \
                    list(self.model.classifier.parameters())
        
        # Add gated fusion params if using gated fusion
        if hasattr(self.model, 'fusion_gate') and self.model.fusion_gate is not None:
            new_params.extend(list(self.model.fusion_gate.parameters()))
        
        self.optimizer = optim.AdamW([
            {'params': pretrained_params, 'lr': lr * 0.1},  # Lower LR for pretrained
            {'params': new_params, 'lr': lr}  # Higher LR for new components
        ], weight_decay=0.01)
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-7
        )
        
        self.scaler = GradScaler() if self.use_amp else None
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            'train_loss': [], 
            'train_acc': [], 
            'val_loss': [], 
            'val_acc': [], 
            'lr': []
        }
        self.best_val_acc = 0.0
        
        print(f"Improved RAG Trainer initialized on {self.device}")
        print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        if self.use_amp:
            print("✓ Mixed precision (FP16) enabled")
    
    def process_batch_with_rag(self, batch):
        """
        Process batch and retrieve knowledge for each question.
        Includes visual-aware retrieval.
        
        Returns:
            Tensors for model forward pass
        """
        images = batch['image'].to(self.device)
        input_ids = batch['question']['input_ids'].to(self.device)
        attention_mask = batch['question']['attention_mask'].to(self.device)
        labels = batch['answer'].to(self.device)
        questions = batch['question_text']
        
        # Get image features for visual-aware retrieval
        # Use vision encoder's CLS token as image representation
        with torch.no_grad():
            vision_outputs = self.model.vision_encoder(pixel_values=images)
            image_features = vision_outputs.last_hidden_state  # [B, num_patches, hidden_dim]
        
        # Retrieve knowledge with visual context
        contexts = []
        for i, question in enumerate(questions):
            img_feat = image_features[i:i+1]  # Keep batch dimension
            retrieved_docs = self.model.retriever.retrieve(
                question, 
                image_features=img_feat
            )
            context = self.model.retriever.format_context(retrieved_docs)
            contexts.append(context if context else "No relevant information available.")
        
        # Tokenize contexts
        context_encoded = self.tokenizer(
            contexts,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        context_input_ids = context_encoded['input_ids'].to(self.device)
        context_attention_mask = context_encoded['attention_mask'].to(self.device)
        
        return images, input_ids, attention_mask, labels, context_input_ids, context_attention_mask
    
    def train_epoch(self, epoch):
        """Train one epoch with gradient accumulation."""
        self.model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        
        self.optimizer.zero_grad()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs} [Train]")
        
        for step, batch in enumerate(pbar):
            # Process batch with RAG and visual-aware retrieval
            images, input_ids, attention_mask, labels, context_input_ids, context_attention_mask = \
                self.process_batch_with_rag(batch)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    logits = self.model(
                        images, input_ids, attention_mask,
                        context_input_ids, context_attention_mask
                    )
                    loss = self.criterion(logits, labels)
                    loss = loss / self.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
            else:
                logits = self.model(
                    images, input_ids, attention_mask,
                    context_input_ids, context_attention_mask
                )
                loss = self.criterion(logits, labels)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
            
            # Gradient accumulation
            if (step + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Metrics
            with torch.no_grad():
                predictions = torch.argmax(logits.detach(), dim=1)
                acc = calculate_accuracy(predictions, labels)
                loss_meter.update(loss.detach().item() * self.gradient_accumulation_steps, images.size(0))
                acc_meter.update(acc, images.size(0))
            
            pbar.set_postfix({
                'loss': loss_meter.avg,
                'acc': acc_meter.avg,
                'lr': self.optimizer.param_groups[1]['lr']  # New layers LR
            })
        
        return loss_meter.avg, acc_meter.avg
    
    @torch.no_grad()
    def evaluate(self):
        """Evaluate on validation set."""
        self.model.eval()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        
        pbar = tqdm(self.val_loader, desc="Evaluating")
        
        for batch in pbar:
            images, input_ids, attention_mask, labels, context_input_ids, context_attention_mask = \
                self.process_batch_with_rag(batch)
            
            logits = self.model(
                images, input_ids, attention_mask,
                context_input_ids, context_attention_mask
            )
            loss = self.criterion(logits, labels)
            
            predictions = torch.argmax(logits, dim=1)
            acc = calculate_accuracy(predictions, labels)
            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(acc, images.size(0))
            
            pbar.set_postfix({'loss': loss_meter.avg, 'acc': acc_meter.avg})
        
        return loss_meter.avg, acc_meter.avg
    
    def train(self):
        """Train model for all epochs."""
        print("\n" + "=" * 60)
        print("Starting Improved RAG-VQA Training")
        print("=" * 60)
        
        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.evaluate()
            self.scheduler.step()
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(self.optimizer.param_groups[1]['lr'])
            
            print(f"\nEpoch {epoch}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"  LR: {self.optimizer.param_groups[1]['lr']:.2e}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, is_best=True)
                print(f"  ✓ Best model saved (val acc: {val_acc:.4f})")
            
            # Save periodic checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(epoch)
        
        print("\n" + "=" * 60)
        print(f"Training complete! Best val acc: {self.best_val_acc:.4f}")
        print("=" * 60)
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_val_acc': self.best_val_acc,
        }
        
        if is_best:
            path = self.checkpoint_dir / 'checkpoint_best.pth'
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        return checkpoint['epoch']


def parse_args():
    parser = argparse.ArgumentParser(description='Train Improved RAG-VQA')
    parser.add_argument('--data_path', type=str, default='data/processed/vqa_rad_closed.csv',
                        help='Path to VQA dataset CSV')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Path to VQA image directory')
    parser.add_argument('--knowledge_base_path', type=str, 
                        default='data/knowledge/medical_kb',
                        help='Path to medical knowledge base')
    parser.add_argument('--batch_size', type=int, default=12,
                        help='Batch size for training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=40,
                        help='Number of epochs')
    parser.add_argument('--checkpoint_dir', type=str, 
                        default='models/checkpoints/rag-improved',
                        help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--use_gated_fusion', action='store_true', default=True,
                        help='Use gated fusion mechanism')
    parser.add_argument('--top_k_docs', type=int, default=3,
                        help='Number of documents to retrieve')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("\n" + "=" * 60)
    print("Improved RAG-VQA Training Configuration")
    print("=" * 60)
    print(f"Data: {args.data_path}")
    print(f"Knowledge Base: {args.knowledge_base_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Gated fusion: {args.use_gated_fusion}")
    print(f"Top-K docs: {args.top_k_docs}")
    print("=" * 60 + "\n")
    
    # Resolve paths
    data_csv = Path(args.data_path)
    image_dir = Path(args.image_dir) if args.image_dir else config.paths.raw_data / "VQA-RAD" / "images"

    # Load dataset
    print("Loading dataset...")
    tokenizer = get_tokenizer(config.model.text_model)
    transform = get_image_transforms(config.model.image_size, is_training=True)
    dataset = MedicalVQADataset(data_csv, image_dir, transform, tokenizer)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"✓ Dataset loaded: {len(dataset)} samples")
    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    # Get number of classes
    num_classes = dataset.get_num_classes()
    print(f"✓ Number of classes: {num_classes}")
    
    # Build knowledge base if not exists
    kb_path = Path(args.knowledge_base_path)
    if not kb_path.exists():
        print("\nBuilding knowledge base...")
        os.system(f"uv run python scripts/build_knowledge_base.py --use_sample --output_dir {args.knowledge_base_path}")
    
    # Create model
    print("\nCreating Improved RAG-VQA model...")
    model = ImprovedRAGVQAModel(
        num_classes=num_classes,
        knowledge_base_path=args.knowledge_base_path,
        top_k_docs=args.top_k_docs,
        use_gated_fusion=args.use_gated_fusion,
    )
    
    # Create trainer
    trainer = ImprovedRAGTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        device=args.device,
        lr=args.learning_rate,
        epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        checkpoint_dir=args.checkpoint_dir,
        use_amp=True,
    )
    
    # Train
    trainer.train()
    
    # Save history
    history_path = Path(args.checkpoint_dir) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(trainer.history, f, indent=2)
    print(f"\n✓ Training history saved to {history_path}")


if __name__ == '__main__':
    main()
