"""Train combined RAG + caption + adversarial-prompting VQA model.

Key ideas:
- Captions are used in two places:
  1) As an explicit modality fused into the VQA head.
  2) As extra query context for retrieval (RAG), via MedicalRetriever(image_caption=...).
- Adversarial prompting is applied by perturbing the *question text* before:
  - tokenization for the model
  - retrieval queries

Example:
  uv run python scripts/train_rag_caption.py \
    --data_path data/processed/vqa_rad_closed_with_captions.csv \
    --image_dir data/raw/VQA-RAD/images \
    --knowledge_base_path data/knowledge/medical_kb \
    --caption_column caption \
    --batch_size 12 \
    --learning_rate 5e-4 \
    --num_epochs 40 \
    --checkpoint_dir models/checkpoints/rag-caption \
    --device cuda \
    --adversarial_prompting \
    --adversarial_probability 0.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
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
from vqa_med.models import RAGCaptionVQAModel
from vqa_med.utils import AverageMeter, calculate_accuracy, get_image_transforms, get_tokenizer
from vqa_med.utils.adversarial import AdversarialPromptConfig, perturb_questions


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
    import pandas as pd

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


class RAGCaptionVQADataset(MedicalVQADataset):
    """Dataset that provides caption tokens + caption text."""

    def __init__(
        self,
        data_file: Path,
        image_dir: Path,
        transform=None,
        tokenizer=None,
        max_length: int = 128,
        caption_column: str = "caption",
        caption_max_length: int = 48,
    ):
        super().__init__(data_file, image_dir, transform, tokenizer, max_length=max_length)
        self.caption_column = caption_column
        self.caption_max_length = caption_max_length

        if self.caption_column not in self.data.columns:
            raise ValueError(
                f"Missing caption column '{self.caption_column}' in {data_file}. "
                "Provide a caption-augmented CSV (or generate captions first)."
            )

    def __getitem__(self, idx: int) -> Dict:
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


class RAGCaptionTrainer:
    def __init__(
        self,
        model: RAGCaptionVQAModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tokenizer,
        device: str,
        lr: float,
        epochs: int,
        checkpoint_dir: Path,
        gradient_accumulation_steps: int = 1,
        use_amp: bool = True,
        top_k_docs: int = 3,
        visual_weight: float = 0.25,
        adversarial_config: Optional[AdversarialPromptConfig] = None,
        context_max_length: int = 256,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.num_epochs = epochs
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.use_amp = use_amp and torch.cuda.is_available()
        self.top_k_docs = top_k_docs
        self.visual_weight = visual_weight
        self.adversarial_config = adversarial_config or AdversarialPromptConfig(enabled=False)
        self.context_max_length = context_max_length

        self.criterion = nn.CrossEntropyLoss()

        pretrained_params = list(self.model.vision_encoder.parameters()) + list(self.model.text_encoder.parameters())
        new_params = (
            list(self.model.context_attention.parameters())
            + list(self.model.vision_attention.parameters())
            + list(self.model.fusion.parameters())
            + list(self.model.classifier.parameters())
        )
        if getattr(self.model, "qc_gate", None) is not None:
            new_params.extend(list(self.model.qc_gate.parameters()))
        if getattr(self.model, "feature_gate", None) is not None:
            new_params.extend(list(self.model.feature_gate.parameters()))

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

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
        }
        self.best_val_acc = 0.0

        print(f"RAG+Caption Trainer initialized on {self.device}")
        print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        if self.adversarial_config.enabled:
            print(
                "Adversarial prompting enabled: "
                f"mode={self.adversarial_config.mode}, "
                f"prob={self.adversarial_config.probability:.2f}, "
                f"seed={self.adversarial_config.seed}"
            )
        if self.use_amp:
            print("✓ Mixed precision (FP16) enabled")

    def _prepare_questions(self, question_texts):
        if isinstance(question_texts, str):
            question_texts = [question_texts]

        if self.adversarial_config.enabled:
            prepared, _ = perturb_questions(question_texts, self.adversarial_config)
            return prepared

        return list(question_texts)

    def _process_batch(self, batch) -> Tuple[torch.Tensor, ...]:
        images = batch["image"].to(self.device)
        labels = batch["answer"].to(self.device)

        # Use question_text so adversarial prompting can be applied.
        prepared_questions = self._prepare_questions(batch["question_text"])
        question_encoded = self.tokenizer(
            prepared_questions,
            max_length=config.model.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        question_ids = question_encoded["input_ids"].to(self.device)
        question_mask = question_encoded["attention_mask"].to(self.device)

        caption_ids = batch["caption"]["input_ids"].to(self.device)
        caption_mask = batch["caption"]["attention_mask"].to(self.device)
        caption_texts = batch.get("caption_text", ["" for _ in prepared_questions])
        if isinstance(caption_texts, str):
            caption_texts = [caption_texts]

        # Extract image features for visual-aware retrieval
        with torch.no_grad():
            vision_outputs = self.model.vision_encoder(pixel_values=images)
            image_features = vision_outputs.last_hidden_state

        contexts = []
        for idx, question in enumerate(prepared_questions):
            img_feat = image_features[idx : idx + 1]
            caption_text = caption_texts[idx] if idx < len(caption_texts) else ""
            retrieved_docs = self.model.retriever.retrieve(
                question,
                image_caption=caption_text,
                image_features=img_feat,
                visual_weight=self.visual_weight,
            )
            context = self.model.retriever.format_context(retrieved_docs)
            contexts.append(context if context else "")

        context_encoded = self.tokenizer(
            contexts,
            max_length=self.context_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        context_ids = context_encoded["input_ids"].to(self.device)
        context_mask = context_encoded["attention_mask"].to(self.device)

        return (
            images,
            question_ids,
            question_mask,
            context_ids,
            context_mask,
            caption_ids,
            caption_mask,
            labels,
        )

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        self.optimizer.zero_grad()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs} [Train]")

        for step, batch in enumerate(pbar):
            (
                images,
                q_ids,
                q_mask,
                ctx_ids,
                ctx_mask,
                cap_ids,
                cap_mask,
                labels,
            ) = self._process_batch(batch)

            if self.use_amp:
                with autocast():
                    logits = self.model(
                        images,
                        q_ids,
                        q_mask,
                        ctx_ids,
                        ctx_mask,
                        cap_ids,
                        cap_mask,
                    )
                    loss = self.criterion(logits, labels)
                    loss = loss / self.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
            else:
                logits = self.model(
                    images,
                    q_ids,
                    q_mask,
                    ctx_ids,
                    ctx_mask,
                    cap_ids,
                    cap_mask,
                )
                loss = self.criterion(logits, labels)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

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

            with torch.no_grad():
                preds = torch.argmax(logits.detach(), dim=1)
                acc = calculate_accuracy(preds, labels)

            loss_meter.update(loss.detach().item() * self.gradient_accumulation_steps, images.size(0))
            acc_meter.update(acc, images.size(0))
            pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}", "acc": f"{acc_meter.avg:.2f}%"})

        return loss_meter.avg, acc_meter.avg

    def validate(self, epoch: int) -> Tuple[float, float]:
        self.model.eval()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.num_epochs} [Val]")

        with torch.no_grad():
            for batch in pbar:
                (
                    images,
                    q_ids,
                    q_mask,
                    ctx_ids,
                    ctx_mask,
                    cap_ids,
                    cap_mask,
                    labels,
                ) = self._process_batch(batch)

                logits = self.model(
                    images,
                    q_ids,
                    q_mask,
                    ctx_ids,
                    ctx_mask,
                    cap_ids,
                    cap_mask,
                )
                loss = self.criterion(logits, labels)

                preds = torch.argmax(logits, dim=1)
                acc = calculate_accuracy(preds, labels)

                loss_meter.update(loss.item(), images.size(0))
                acc_meter.update(acc, images.size(0))
                pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}", "acc": f"{acc_meter.avg:.2f}%"})

        return loss_meter.avg, acc_meter.avg

    def save_checkpoint(self, epoch: int, val_acc: float, is_best: bool):
        payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_acc": val_acc,
            "history": self.history,
        }
        latest_path = self.checkpoint_dir / "checkpoint_latest.pth"
        torch.save(payload, latest_path)

        if is_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pth"
            torch.save(payload, best_path)
            print(f"✓ Best checkpoint saved: {best_path} (val_acc={val_acc:.2f}%)")

    def train(self):
        print("\n" + "=" * 60)
        print("Training RAG+Caption VQA Model")
        print("=" * 60)

        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)
            self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc

            self.save_checkpoint(epoch, val_acc, is_best)

            print(
                f"Epoch {epoch}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%, lr={current_lr:.2e}"
            )

        with open(self.checkpoint_dir / "training_history.json", "w") as handle:
            json.dump(self.history, handle, indent=2)

        print("\n" + "=" * 60)
        print("✓ Training completed")
        print("=" * 60)
        print(f"Best val acc: {self.best_val_acc:.2f}%")


def parse_args():
    parser = argparse.ArgumentParser(description="Train RAG+Caption VQA model")
    parser.add_argument("--data_path", type=str, default=str(config.paths.processed_data / "vqa_rad_closed.csv"))
    parser.add_argument("--image_dir", type=str, default=str(config.paths.raw_data / "VQA-RAD" / "images"))
    parser.add_argument("--knowledge_base_path", type=str, required=True)

    parser.add_argument("--caption_column", type=str, default="caption")
    parser.add_argument("--caption_max_length", type=int, default=48)
    parser.add_argument("--generate_captions", action="store_true")
    parser.add_argument("--caption_cache_file", type=str, default=None)
    parser.add_argument(
        "--caption_model_name",
        type=str,
        default="Salesforce/blip-image-captioning-base",
    )
    parser.add_argument("--caption_batch_size", type=int, default=8)

    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--top_k_docs", type=int, default=3)
    parser.add_argument("--visual_weight", type=float, default=0.25)
    parser.add_argument("--context_max_length", type=int, default=256)
    parser.add_argument("--use_query_expansion", action="store_true")

    parser.add_argument("--checkpoint_dir", type=str, default=str(config.paths.checkpoints_dir / "rag-caption"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--no_amp", action="store_true")

    # Adversarial prompting
    parser.add_argument("--adversarial_prompting", action="store_true")
    parser.add_argument("--adversarial_probability", type=float, default=0.5)
    parser.add_argument("--adversarial_mode", type=str, default="mixed")
    parser.add_argument("--adversarial_seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_path = Path(args.data_path)
    image_dir = Path(args.image_dir)

    if args.generate_captions:
        cache_path = (
            Path(args.caption_cache_file)
            if args.caption_cache_file
            else config.paths.processed_data / "vqa_rad_with_captions.csv"
        )
        data_path = build_or_load_caption_cache(
            data_csv=data_path,
            image_dir=image_dir,
            caption_cache_file=cache_path,
            caption_column=args.caption_column,
            caption_model_name=args.caption_model_name,
            batch_size=args.caption_batch_size,
            device=args.device,
        )

    tokenizer = get_tokenizer(config.model.text_model)
    transform = get_image_transforms(config.model.image_size, is_training=True)

    dataset = RAGCaptionVQADataset(
        data_file=data_path,
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
    train_set, val_set, _ = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = RAGCaptionVQAModel(
        num_classes=num_classes,
        knowledge_base_path=str(args.knowledge_base_path),
        vision_model_name=config.model.vision_model,
        text_model_name=config.model.text_model,
        hidden_dim=config.model.hidden_dim,
        dropout=config.model.dropout,
        num_attention_heads=8,
        top_k_docs=args.top_k_docs,
    )

    if args.use_query_expansion:
        model.retriever.use_query_expansion = True

    adversarial_cfg = AdversarialPromptConfig(
        enabled=args.adversarial_prompting,
        probability=args.adversarial_probability,
        mode=args.adversarial_mode,
        seed=args.adversarial_seed,
    )

    trainer = RAGCaptionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        device=args.device,
        lr=args.learning_rate,
        epochs=args.num_epochs,
        checkpoint_dir=Path(args.checkpoint_dir),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_amp=not args.no_amp,
        top_k_docs=args.top_k_docs,
        visual_weight=args.visual_weight,
        adversarial_config=adversarial_cfg,
        context_max_length=args.context_max_length,
    )

    trainer.train()


if __name__ == "__main__":
    main()
