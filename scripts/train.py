"""
Training script for Medical VQA model.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import argparse

from vqa_med.models import BaseVQAModel, VQAModelWrapper
from vqa_med.data import MedicalVQADataset, MedicalVQADataModule
from vqa_med.utils import (
    get_image_transforms,
    get_tokenizer,
    AverageMeter,
    calculate_accuracy
)
from vqa_med.utils.adversarial import AdversarialPromptConfig, perturb_questions
from vqa_med.config import config


class VQATrainer:
    """Trainer class for VQA model."""
    
    def __init__(
        self,
        model: BaseVQAModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
        learning_rate: float = 1e-4,
        num_epochs: int = 10,
        checkpoint_dir: Path = None,
        tokenizer=None,
        adversarial_prompting: bool = False,
        adversarial_probability: float = 0.5,
        adversarial_mode: str = "mixed",
        adversarial_seed: int = 42,
    ):
        """
        Args:
            model: VQA model instance
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            checkpoint_dir: Directory to save checkpoints
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.tokenizer = tokenizer
        self.max_text_length = config.model.max_text_length
        self.adversarial_config = AdversarialPromptConfig(
            enabled=adversarial_prompting,
            probability=adversarial_probability,
            mode=adversarial_mode,
            seed=adversarial_seed,
        )
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
                # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=2,
        )
        self.current_lr = learning_rate  # Track current LR
        
        # Checkpoint directory
        self.checkpoint_dir = checkpoint_dir or config.paths.checkpoints_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        self.best_val_acc = 0.0
        
        print(f"Trainer initialized on device: {self.device}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        if self.adversarial_config.enabled:
            print(
                "Adversarial prompting enabled: "
                f"mode={self.adversarial_config.mode}, "
                f"prob={self.adversarial_config.probability:.2f}, "
                f"seed={self.adversarial_config.seed}"
            )

    def _prepare_question_batch(self, batch, apply_adversarial: bool = False):
        question_texts = batch["question_text"]
        if isinstance(question_texts, str):
            question_texts = [question_texts]

        if apply_adversarial and self.adversarial_config.enabled and self.tokenizer is not None:
            prepared_questions, prompt_flags = perturb_questions(question_texts, self.adversarial_config)
            encoded = self.tokenizer(
                prepared_questions,
                max_length=self.max_text_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
            return encoded['input_ids'].to(self.device), encoded['attention_mask'].to(self.device), prompt_flags

        return (
            batch['question']['input_ids'].to(self.device),
            batch['question']['attention_mask'].to(self.device),
            [False for _ in question_texts],
        )
    
    def train_epoch(self, epoch: int) -> tuple:
        """Train for one epoch."""
        self.model.train()
        
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch['image'].to(self.device)
            input_ids, attention_mask, _ = self._prepare_question_batch(batch, apply_adversarial=True)
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
            
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=1)
            acc = calculate_accuracy(predictions, labels)
            
            # Update meters
            batch_size = images.size(0)
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(acc, batch_size)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'acc': f'{acc_meter.avg:.2f}%'
            })
        
        return loss_meter.avg, acc_meter.avg
    
    def validate(self, epoch: int) -> tuple:
        """Validate the model."""
        self.model.eval()
        
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.num_epochs} [Val]")
        
        with torch.no_grad():
            for batch in pbar:
                # Move to device
                images = batch['image'].to(self.device)
                input_ids, attention_mask, _ = self._prepare_question_batch(batch, apply_adversarial=False)
                labels = batch['answer'].to(self.device)
                
                # Forward pass
                logits = self.model(images, input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                
                # Calculate accuracy
                predictions = torch.argmax(logits, dim=1)
                acc = calculate_accuracy(predictions, labels)
                
                # Update meters
                batch_size = images.size(0)
                loss_meter.update(loss.item(), batch_size)
                acc_meter.update(acc, batch_size)
                
                # Update progress bar
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
            'history': self.history,
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "checkpoint_latest.pth"
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pth"
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved (acc: {val_acc:.2f}%)")
        
        # Save epoch checkpoint
        epoch_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load checkpoint and resume training."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        
        start_epoch = checkpoint['epoch'] + 1
        print(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
        return start_epoch
    
    def train(self, resume_from: Path = None):
        """Full training loop."""
        start_epoch = 1
        
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)
        
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)
        
        for epoch in range(start_epoch, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            print("-" * 60)
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            
            # Update learning rate
                        # Update learning rate
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Print if LR changed
            if current_lr != old_lr:
                print(f"Learning rate reduced: {old_lr:.2e} → {current_lr:.2e}")
                        
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # Print epoch summary
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
        
        # Save training history
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to: {history_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Medical VQA Model')
    
    # Data arguments
    parser.add_argument('--data_csv', type=str, default=None,
                        help='Path to CSV file with QA pairs')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Directory containing images')
    parser.add_argument('--data_type', type=str, default='full',
                        choices=['closed', 'full'],
                        help='Use closed-ended or full dataset (default: full for all answer types)')
    
    # Model arguments
    parser.add_argument('--vision_model', type=str, default=None,
                        help='Vision model name')
    parser.add_argument('--text_model', type=str, default=None,
                        help='Text model name')
    parser.add_argument('--hidden_dim', type=int, default=None,
                        help='Hidden dimension size')
    parser.add_argument('--dropout', type=float, default=None,
                        help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of data loading workers')
    
    # Other arguments
    parser.add_argument('--device', type=str, default=None,
                        choices=['cuda', 'cpu'],
                        help='Device to train on')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory to save checkpoints')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--adversarial_prompting', action='store_true',
                        help='Enable adversarial prompt augmentation during training')
    parser.add_argument('--adversarial_probability', type=float, default=0.5,
                        help='Probability of perturbing each training question')
    parser.add_argument('--adversarial_mode', type=str, default='mixed',
                        choices=['mixed', 'instruction', 'careful', 'strict', 'contrast'],
                        help='Prompt style used for adversarial augmentation')
    parser.add_argument('--adversarial_seed', type=int, default=42,
                        help='Seed for deterministic adversarial prompt selection')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    print("=" * 60)
    print("Medical VQA Training")
    print("=" * 60)
    
    # Data paths
    if args.data_csv:
        data_csv = Path(args.data_csv)
    else:
        data_csv = config.paths.processed_data / f"vqa_rad_{args.data_type}.csv"
    
    if args.image_dir:
        image_dir = Path(args.image_dir)
    else:
        image_dir = config.paths.raw_data / "VQA-RAD" / "images"
    
    # Check if data exists
    if not data_csv.exists():
        print(f"\nERROR: Data file not found at {data_csv}")
        print("Please run: uv run python scripts/prepare_vqa_rad.py")
        return
    
    if not image_dir.exists():
        print(f"\nERROR: Image directory not found at {image_dir}")
        print("Please download VQA-RAD dataset to data/raw/VQA-RAD/")
        return
    
    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Data CSV: {data_csv}")
    print(f"  Image Dir: {image_dir}")
    print(f"  Batch Size: {args.batch_size or config.model.batch_size}")
    print(f"  Learning Rate: {args.learning_rate or config.model.learning_rate}")
    print(f"  Epochs: {args.num_epochs or config.model.num_epochs}")
    print(f"  Device: {args.device or config.model.device}")
    
    # Prepare data
    print("\nPreparing data...")
    
    # Get transforms and tokenizer
    train_transform = get_image_transforms(
        image_size=config.model.image_size,
        is_training=True
    )
    val_transform = get_image_transforms(
        image_size=config.model.image_size,
        is_training=False
    )
    tokenizer = get_tokenizer(args.text_model or config.model.text_model)
    
    # Create data module
    data_module = MedicalVQADataModule(
        data_file=data_csv,
        image_dir=image_dir,
        transform=train_transform,
        tokenizer=tokenizer,
        batch_size=args.batch_size or config.model.batch_size,
        num_workers=args.num_workers or config.data.num_workers,
        train_split=config.data.train_split,
        val_split=config.data.val_split,
        test_split=config.data.test_split,
    )
    
    # Setup data
    num_classes = data_module.setup()
    
    # Get dataloaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    # Create model
    print("\nInitializing model...")
    model = BaseVQAModel(
        num_classes=num_classes,
        vision_model_name=args.vision_model or config.model.vision_model,
        text_model_name=args.text_model or config.model.text_model,
        hidden_dim=args.hidden_dim or config.model.hidden_dim,
        dropout=args.dropout or config.model.dropout,
    )
    
    # Checkpoint directory
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
    else:
        checkpoint_dir = config.paths.checkpoints_dir
    
    # Create trainer
    trainer = VQATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device or config.model.device,
        learning_rate=args.learning_rate or config.model.learning_rate,
        num_epochs=args.num_epochs or config.model.num_epochs,
        checkpoint_dir=checkpoint_dir,
        tokenizer=tokenizer,
        adversarial_prompting=args.adversarial_prompting,
        adversarial_probability=args.adversarial_probability,
        adversarial_mode=args.adversarial_mode,
        adversarial_seed=args.adversarial_seed,
    )
    
    # Train
    resume_from = Path(args.resume_from) if args.resume_from else None
    trainer.train(resume_from=resume_from)


if __name__ == "__main__":
    main()