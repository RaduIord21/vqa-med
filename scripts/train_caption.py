"""Train caption-augmented VQA model with optional BLIP caption generation."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor

from vqa_med.config import config
from vqa_med.data import MedicalVQADataset
from vqa_med.models import CaptionVQAModel
from vqa_med.utils import AverageMeter, calculate_accuracy, get_image_transforms, get_tokenizer


class CaptionVQADataset(MedicalVQADataset):
    """Dataset that adds tokenized captions in addition to questions."""

    def __init__(
        self,
        data_file: Path,
        image_dir: Path,
        transform=None,
        tokenizer=None,
        max_length: int = 128,
        caption_column: str = "caption",
        caption_max_length: int = 64,
    ):
        super().__init__(data_file, image_dir, transform, tokenizer, max_length=max_length)
        self.caption_column = caption_column
        self.caption_max_length = caption_max_length

        if self.caption_column not in self.data.columns:
            raise ValueError(
                f"Missing caption column '{self.caption_column}' in {data_file}. "
                "Generate captions first or pass a CSV with captions."
            )

    def __getitem__(self, idx: int):
        sample = super().__getitem__(idx)
        caption_text = str(self.data.iloc[idx][self.caption_column])

        caption_encoded = self.tokenizer(
            caption_text,
            max_length=self.caption_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        sample["caption"] = {
            "input_ids": caption_encoded["input_ids"].squeeze(0),
            "attention_mask": caption_encoded["attention_mask"].squeeze(0),
        }
        sample["caption_text"] = caption_text
        return sample


def build_or_load_caption_cache(
    data_csv: Path,
    image_dir: Path,
    caption_cache_file: Path,
    caption_column: str,
    caption_model_name: str,
    batch_size: int,
    device: str,
) -> Path:
    """Generate per-image captions once and write an augmented CSV."""
    df = pd.read_csv(data_csv)
    if caption_column in df.columns and df[caption_column].notna().all():
        print(f"Caption column '{caption_column}' already present in {data_csv}. Skipping generation.")
        return data_csv

    caption_cache_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading caption model: {caption_model_name}")
    processor = BlipProcessor.from_pretrained(caption_model_name)
    caption_model = BlipForConditionalGeneration.from_pretrained(caption_model_name)

    run_device = torch.device(device if torch.cuda.is_available() else "cpu")
    caption_model = caption_model.to(run_device)
    caption_model.eval()

    image_paths = df["image_path"].dropna().unique().tolist()
    captions_by_image = {}

    pbar = tqdm(range(0, len(image_paths), batch_size), desc="Generating captions")
    for start in pbar:
        batch_paths = image_paths[start : start + batch_size]
        images = []
        valid_paths = []

        for rel_path in batch_paths:
            full_path = image_dir / rel_path
            try:
                images.append(Image.open(full_path).convert("RGB"))
                valid_paths.append(rel_path)
            except Exception:
                continue

        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt").to(run_device)
        with torch.no_grad():
            generated = caption_model.generate(**inputs, max_new_tokens=40)

        decoded = processor.batch_decode(generated, skip_special_tokens=True)
        for rel_path, caption in zip(valid_paths, decoded):
            captions_by_image[rel_path] = caption.strip()

    df[caption_column] = df["image_path"].map(captions_by_image).fillna("")
    df.to_csv(caption_cache_file, index=False)
    print(f"Saved caption-augmented CSV: {caption_cache_file}")
    return caption_cache_file


class CaptionTrainer:
    """Trainer for caption-augmented VQA model."""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        lr,
        epochs,
        checkpoint_dir,
        use_amp=True,
        gradient_accumulation_steps=1,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp and torch.cuda.is_available()
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = epochs

        self.criterion = nn.CrossEntropyLoss()

        pretrained_params = list(self.model.vision_encoder.parameters()) + list(
            self.model.text_encoder.parameters()
        )
        new_params = list(self.model.cross_attention.parameters()) + list(
            self.model.feature_gate.parameters()
        ) + list(self.model.fusion.parameters()) + list(self.model.classifier.parameters())

        self.optimizer = optim.AdamW(
            [
                {"params": pretrained_params, "lr": lr * 0.1},
                {"params": new_params, "lr": lr},
            ],
            weight_decay=0.01,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-7
        )

        self.scaler = GradScaler() if self.use_amp else None
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
        self.best_val_acc = 0.0

    def _forward_batch(self, batch):
        images = batch["image"].to(self.device)
        q_ids = batch["question"]["input_ids"].to(self.device)
        q_mask = batch["question"]["attention_mask"].to(self.device)
        c_ids = batch["caption"]["input_ids"].to(self.device)
        c_mask = batch["caption"]["attention_mask"].to(self.device)
        labels = batch["answer"].to(self.device)

        logits = self.model(images, q_ids, q_mask, c_ids, c_mask)
        return logits, labels, images.size(0)

    def train_epoch(self, epoch):
        self.model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs} [Train]")
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            if self.use_amp:
                with autocast():
                    logits, labels, batch_size = self._forward_batch(batch)
                    loss = self.criterion(logits, labels) / self.gradient_accumulation_steps
                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                logits, labels, batch_size = self._forward_batch(batch)
                loss = self.criterion(logits, labels) / self.gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            predictions = torch.argmax(logits, dim=1)
            acc = calculate_accuracy(predictions, labels)

            loss_meter.update(loss.item() * self.gradient_accumulation_steps, batch_size)
            acc_meter.update(acc, batch_size)

            pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}", "acc": f"{acc_meter.avg:.2f}%"})

        return loss_meter.avg, acc_meter.avg

    def validate(self, epoch):
        self.model.eval()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.num_epochs} [Val]")

        with torch.no_grad():
            for batch in pbar:
                if self.use_amp:
                    with autocast():
                        logits, labels, batch_size = self._forward_batch(batch)
                        loss = self.criterion(logits, labels)
                else:
                    logits, labels, batch_size = self._forward_batch(batch)
                    loss = self.criterion(logits, labels)

                predictions = torch.argmax(logits, dim=1)
                acc = calculate_accuracy(predictions, labels)

                loss_meter.update(loss.item(), batch_size)
                acc_meter.update(acc, batch_size)
                pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}", "acc": f"{acc_meter.avg:.2f}%"})

        return loss_meter.avg, acc_meter.avg

    def save_checkpoint(self, epoch, val_acc, is_best):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_acc": val_acc,
            "history": self.history,
        }

        torch.save(checkpoint, self.checkpoint_dir / "checkpoint_latest.pth")

        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "checkpoint_best.pth")
            print(f"Best model updated: {val_acc:.2f}%")

    def train(self):
        print("\n" + "=" * 60)
        print("Caption-Augmented VQA Training")
        print("=" * 60)

        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)

            self.scheduler.step()
            lr = self.optimizer.param_groups[1]["lr"]

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(lr)

            print(
                f"Epoch {epoch}: train {train_loss:.4f}/{train_acc:.2f}% | "
                f"val {val_loss:.4f}/{val_acc:.2f}% | lr {lr:.2e}"
            )

            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            self.save_checkpoint(epoch, val_acc, is_best)

        print(f"\nBest Validation Accuracy: {self.best_val_acc:.2f}%")

        with open(self.checkpoint_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--caption_column", type=str, default="caption")
    parser.add_argument("--caption_max_length", type=int, default=64)
    parser.add_argument("--generate_captions", action="store_true")
    parser.add_argument("--caption_cache_file", type=str, default=None)
    parser.add_argument("--caption_model_name", type=str, default="Salesforce/blip-image-captioning-base")
    parser.add_argument("--caption_batch_size", type=int, default=8)

    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_csv = Path(args.data_csv) if args.data_csv else config.paths.processed_data / "vqa_rad_closed.csv"
    image_dir = Path(args.image_dir) if args.image_dir else config.paths.raw_data / "VQA-RAD" / "images"

    if args.generate_captions:
        cache_path = (
            Path(args.caption_cache_file)
            if args.caption_cache_file
            else config.paths.processed_data / "vqa_rad_closed_with_captions.csv"
        )
        data_csv = build_or_load_caption_cache(
            data_csv=data_csv,
            image_dir=image_dir,
            caption_cache_file=cache_path,
            caption_column=args.caption_column,
            caption_model_name=args.caption_model_name,
            batch_size=args.caption_batch_size,
            device=args.device,
        )

    transform = get_image_transforms(config.model.image_size, is_training=True)
    tokenizer = get_tokenizer(config.model.text_model)

    dataset = CaptionVQADataset(
        data_file=data_csv,
        image_dir=image_dir,
        transform=transform,
        tokenizer=tokenizer,
        max_length=config.model.max_text_length,
        caption_column=args.caption_column,
        caption_max_length=args.caption_max_length,
    )

    num_classes = dataset.get_num_classes()
    total = len(dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size

    train_ds, val_ds, _ = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    num_workers = config.data.num_workers
    use_workers = num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=use_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=use_workers,
    )

    print(f"Dataset: {total} samples (Train: {train_size}, Val: {val_size}, Test: {test_size})")
    print(f"Classes: {num_classes}")

    model = CaptionVQAModel(
        num_classes=num_classes,
        num_attention_heads=args.num_attention_heads,
        dropout=0.2,
    )

    checkpoint_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else config.paths.checkpoints_dir / "caption_vqa"
    )

    trainer = CaptionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        lr=args.learning_rate,
        epochs=args.num_epochs,
        checkpoint_dir=checkpoint_dir,
        use_amp=not args.no_amp,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    trainer.train()


if __name__ == "__main__":
    main()
