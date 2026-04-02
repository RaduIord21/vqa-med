"""Reusable benchmarking utilities for VQA models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score

from vqa_med.config import config
from vqa_med.data import MedicalVQADataset
from vqa_med.models import (
    BaseVQAModel,
    CaptionVQAModel,
    ImprovedRAGVQAModel,
    VQAModelWrapper,
)
from vqa_med.utils import get_image_transforms, get_tokenizer


@dataclass
class BenchmarkSpec:
    """Configuration for one benchmarked model."""

    name: str
    model_type: str
    checkpoint_path: Optional[Path]
    data_csv: Path
    image_dir: Path
    caption_column: str = "caption"
    caption_max_length: int = 48
    knowledge_base_path: Optional[Path] = None
    batch_size: int = 12
    device: str = "cuda"
    seed: int = 42
    split: str = "val"
    top_k_docs: int = 3
    available: bool = True
    notes: str = ""


@dataclass
class BenchmarkResult:
    """Output of one benchmark run."""

    name: str
    model_type: str
    status: str
    accuracy: Optional[float]
    macro_f1: Optional[float]
    weighted_f1: Optional[float]
    num_samples: int
    checkpoint_path: Optional[str]
    inference_path: Optional[str] = None
    confusion_matrix_path: Optional[str] = None
    notes: str = ""


class CaptionVQADataset(MedicalVQADataset):
    """MedicalVQADataset with tokenized captions."""

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
                f"Missing caption column '{self.caption_column}' in {data_file}."
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


class BenchmarkRunner:
    """Run a consistent benchmark across VQA model variants."""

    def __init__(self, specs: Sequence[BenchmarkSpec], output_dir: Optional[Path] = None):
        self.specs = list(specs)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.tokenizer = get_tokenizer(config.model.text_model)

    def run(self) -> List[BenchmarkResult]:
        results: List[BenchmarkResult] = []
        for spec in self.specs:
            results.append(self._run_one(spec))
        return results

    def save_report(self, results: Sequence[BenchmarkResult], output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir

        payload = [asdict(result) for result in results]
        with open(output_dir / "benchmark_results.json", "w") as handle:
            json.dump(payload, handle, indent=2)

        frame = pd.DataFrame(payload)
        frame.to_csv(output_dir / "benchmark_results.csv", index=False)
        return output_dir

    def _run_one(self, spec: BenchmarkSpec) -> BenchmarkResult:
        if not spec.available:
            return BenchmarkResult(
                name=spec.name,
                model_type=spec.model_type,
                status="skipped",
                accuracy=None,
                macro_f1=None,
                weighted_f1=None,
                num_samples=0,
                checkpoint_path=str(spec.checkpoint_path) if spec.checkpoint_path else None,
                inference_path=None,
                confusion_matrix_path=None,
                notes=spec.notes or "Model marked unavailable.",
            )

        if spec.checkpoint_path is None:
            return BenchmarkResult(
                name=spec.name,
                model_type=spec.model_type,
                status="skipped",
                accuracy=None,
                macro_f1=None,
                weighted_f1=None,
                num_samples=0,
                checkpoint_path=None,
                inference_path=None,
                confusion_matrix_path=None,
                notes="Missing checkpoint path.",
            )

        checkpoint_path = Path(spec.checkpoint_path)
        if not checkpoint_path.exists():
            return BenchmarkResult(
                name=spec.name,
                model_type=spec.model_type,
                status="skipped",
                accuracy=None,
                macro_f1=None,
                weighted_f1=None,
                num_samples=0,
                checkpoint_path=str(checkpoint_path),
                inference_path=None,
                confusion_matrix_path=None,
                notes="Checkpoint file not found.",
            )

        dataset = self._build_dataset(spec)
        if len(dataset) == 0:
            return BenchmarkResult(
                name=spec.name,
                model_type=spec.model_type,
                status="skipped",
                accuracy=None,
                macro_f1=None,
                weighted_f1=None,
                num_samples=0,
                checkpoint_path=str(checkpoint_path),
                inference_path=None,
                confusion_matrix_path=None,
                notes="Dataset is empty.",
            )

        train_subset, val_subset, test_subset = self._split_dataset(dataset, spec.seed)
        eval_subset = self._select_subset(train_subset, val_subset, test_subset, spec.split)

        if eval_subset is None:
            return BenchmarkResult(
                name=spec.name,
                model_type=spec.model_type,
                status="skipped",
                accuracy=None,
                macro_f1=None,
                weighted_f1=None,
                num_samples=0,
                checkpoint_path=str(checkpoint_path),
                inference_path=None,
                confusion_matrix_path=None,
                notes=f"Unsupported split '{spec.split}'.",
            )

        try:
            _, predictor = self._build_predictor(spec, dataset.get_num_classes(), checkpoint_path)
            accuracy, macro_f1, weighted_f1, sample_count, inference_path, confusion_path = self._evaluate(
                spec,
                eval_subset,
                predictor,
                checkpoint_path,
            )
        except Exception as exc:
            return BenchmarkResult(
                name=spec.name,
                model_type=spec.model_type,
                status="error",
                accuracy=None,
                macro_f1=None,
                weighted_f1=None,
                num_samples=0,
                checkpoint_path=str(checkpoint_path),
                inference_path=None,
                confusion_matrix_path=None,
                notes=f"{type(exc).__name__}: {exc}",
            )

        return BenchmarkResult(
            name=spec.name,
            model_type=spec.model_type,
            status="ok",
            accuracy=accuracy,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            num_samples=sample_count,
            checkpoint_path=str(checkpoint_path),
            inference_path=inference_path,
            confusion_matrix_path=confusion_path,
            notes=spec.notes,
        )

    def _build_dataset(self, spec: BenchmarkSpec):
        transform = get_image_transforms(config.model.image_size, is_training=False)
        data_csv = Path(spec.data_csv)
        image_dir = Path(spec.image_dir)

        if spec.model_type == "caption":
            return CaptionVQADataset(
                data_file=data_csv,
                image_dir=image_dir,
                transform=transform,
                tokenizer=self.tokenizer,
                max_length=config.model.max_text_length,
                caption_column=spec.caption_column,
                caption_max_length=spec.caption_max_length,
            )

        return MedicalVQADataset(
            data_file=data_csv,
            image_dir=image_dir,
            transform=transform,
            tokenizer=self.tokenizer,
            max_length=config.model.max_text_length,
        )

    @staticmethod
    def _split_dataset(dataset, seed: int):
        total = len(dataset)
        train_size = int(0.7 * total)
        val_size = int(0.15 * total)
        test_size = total - train_size - val_size
        return random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(seed),
        )

    @staticmethod
    def _select_subset(train_subset, val_subset, test_subset, split: str):
        normalized = split.lower()
        if normalized == "train":
            return train_subset
        if normalized == "val":
            return val_subset
        if normalized == "test":
            return test_subset
        return None

    def _build_predictor(
        self,
        spec: BenchmarkSpec,
        num_classes: int,
        checkpoint_path: Path,
    ) -> Tuple[torch.nn.Module, Callable]:
        device = torch.device(spec.device if torch.cuda.is_available() else "cpu")

        if spec.model_type == "baseline":
            model = BaseVQAModel(num_classes=num_classes)
            wrapper = VQAModelWrapper(model, device=spec.device)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            wrapper.model.load_state_dict(state_dict)
            return wrapper.model, lambda batch: self._predict_baseline(wrapper.model, batch, device)

        if spec.model_type == "rag":
            if spec.knowledge_base_path is None:
                raise ValueError("knowledge_base_path is required for RAG benchmarking.")
            model = ImprovedRAGVQAModel(
                num_classes=num_classes,
                knowledge_base_path=str(spec.knowledge_base_path),
                top_k_docs=spec.top_k_docs,
            )
            model.to(device)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict)
            return model, lambda batch: self._predict_rag(model, batch, device)

        if spec.model_type == "caption":
            model = CaptionVQAModel(num_classes=num_classes)
            model.to(device)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict)
            return model, lambda batch: self._predict_caption(model, batch, device)

        if spec.model_type == "kg":
            raise NotImplementedError(
                "KG-enhanced model is not implemented in this repository yet."
            )

        raise ValueError(f"Unsupported model_type: {spec.model_type}")

    def _evaluate(self, spec: BenchmarkSpec, dataset, predictor: Callable, checkpoint_path: Path):
        loader = DataLoader(
            dataset,
            batch_size=spec.batch_size,
            shuffle=False,
            num_workers=0,
        )

        correct = 0
        total = 0
        all_predictions: List[int] = []
        all_labels: List[int] = []
        all_prediction_texts: List[str] = []
        all_label_texts: List[str] = []
        all_questions: List[str] = []

        base_dataset = dataset.dataset if hasattr(dataset, "dataset") else dataset
        idx_to_answer = base_dataset.idx_to_answer
        class_labels = [idx_to_answer[idx] for idx in range(len(idx_to_answer))]

        for batch in tqdm(loader, desc=f"Benchmarking {spec.name}"):
            predictions, labels = predictor(batch)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            all_predictions.extend(predictions.tolist())
            all_labels.extend(labels.tolist())
            all_prediction_texts.extend([idx_to_answer[idx] for idx in predictions.tolist()])
            all_label_texts.extend(batch["answer_text"])
            all_questions.extend(batch["question_text"])

        accuracy = 100.0 * correct / total if total else 0.0
        macro_f1 = f1_score(all_labels, all_predictions, average="macro", zero_division=0) if total else None
        weighted_f1 = f1_score(all_labels, all_predictions, average="weighted", zero_division=0) if total else None

        confusion = confusion_matrix(all_labels, all_predictions) if total else np.zeros((0, 0), dtype=int)
        inference_path, confusion_path = self._save_inference_artifacts(
            spec=spec,
            checkpoint_path=checkpoint_path,
            questions=all_questions,
            label_indices=all_labels,
            predicted_indices=all_predictions,
            label_texts=all_label_texts,
            predicted_texts=all_prediction_texts,
            confusion=confusion,
            class_labels=class_labels,
        )

        return accuracy, macro_f1, weighted_f1, total, inference_path, confusion_path

    def _save_inference_artifacts(
        self,
        spec: BenchmarkSpec,
        checkpoint_path: Path,
        questions: List[str],
        label_indices: List[int],
        predicted_indices: List[int],
        label_texts: List[str],
        predicted_texts: List[str],
        confusion: np.ndarray,
        class_labels: List[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        if self.output_dir is None:
            return None, None

        model_dir = self.output_dir / spec.name
        model_dir.mkdir(parents=True, exist_ok=True)

        inference_df = pd.DataFrame(
            {
                "question": questions,
                "ground_truth_label": label_indices,
                "ground_truth_text": label_texts,
                "predicted_label": predicted_indices,
                "predicted_text": predicted_texts,
                "correct": [p == l for p, l in zip(predicted_indices, label_indices)],
            }
        )
        inference_path = model_dir / "inference.csv"
        inference_df.to_csv(inference_path, index=False)

        confusion_path = model_dir / "confusion_matrix.json"
        with open(confusion_path, "w") as handle:
            json.dump(
                {
                    "labels": class_labels,
                    "matrix": confusion.tolist(),
                },
                handle,
                indent=2,
            )

        return str(inference_path), str(confusion_path)

    @staticmethod
    def _predict_baseline(model, batch, device):
        images = batch["image"].to(device)
        input_ids = batch["question"]["input_ids"].to(device)
        attention_mask = batch["question"]["attention_mask"].to(device)
        labels = batch["answer"].to(device)
        logits = model(images, input_ids, attention_mask)
        predictions = torch.argmax(logits, dim=1).cpu()
        return predictions, labels.cpu()

    def _predict_caption(self, model, batch, device):
        images = batch["image"].to(device)
        question_ids = batch["question"]["input_ids"].to(device)
        question_mask = batch["question"]["attention_mask"].to(device)
        caption_ids = batch["caption"]["input_ids"].to(device)
        caption_mask = batch["caption"]["attention_mask"].to(device)
        labels = batch["answer"].to(device)
        logits = model(images, question_ids, question_mask, caption_ids, caption_mask)
        predictions = torch.argmax(logits, dim=1).cpu()
        return predictions, labels.cpu()

    def _predict_rag(self, model, batch, device):
        images = batch["image"].to(device)
        input_ids = batch["question"]["input_ids"].to(device)
        attention_mask = batch["question"]["attention_mask"].to(device)
        labels = batch["answer"].to(device)

        questions = batch["question_text"]
        if isinstance(questions, str):
            questions = [questions]

        contexts = []
        for question in questions:
            retrieved_docs = model.retrieve_knowledge([question])
            contexts.append(retrieved_docs[0] if retrieved_docs else "")

        context_encoded = self.tokenizer(
            contexts,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        context_input_ids = context_encoded["input_ids"].to(device)
        context_attention_mask = context_encoded["attention_mask"].to(device)
        logits = model(
            images,
            input_ids,
            attention_mask,
            context_input_ids,
            context_attention_mask,
        )
        predictions = torch.argmax(logits, dim=1).cpu()
        return predictions, labels.cpu()


def build_default_specs(
    data_csv: Path,
    image_dir: Path,
    baseline_checkpoint: Optional[Path],
    kg_checkpoint: Optional[Path],
    rag_checkpoint: Optional[Path],
    caption_checkpoint: Optional[Path],
    knowledge_base_path: Optional[Path],
    caption_column: str,
    caption_max_length: int,
    batch_size: int,
    device: str,
    seed: int,
    split: str,
    top_k_docs: int,
) -> List[BenchmarkSpec]:
    return [
        BenchmarkSpec(
            name="baseline",
            model_type="baseline",
            checkpoint_path=baseline_checkpoint,
            data_csv=data_csv,
            image_dir=image_dir,
            batch_size=batch_size,
            device=device,
            seed=seed,
            split=split,
            notes="Original baseline model.",
        ),
        BenchmarkSpec(
            name="kg-enhanced",
            model_type="kg",
            checkpoint_path=kg_checkpoint,
            data_csv=data_csv,
            image_dir=image_dir,
            batch_size=batch_size,
            device=device,
            seed=seed,
            split=split,
            available=False,
            notes="KG-enhanced model is not implemented yet.",
        ),
        BenchmarkSpec(
            name="rag",
            model_type="rag",
            checkpoint_path=rag_checkpoint,
            data_csv=data_csv,
            image_dir=image_dir,
            knowledge_base_path=knowledge_base_path,
            batch_size=batch_size,
            device=device,
            seed=seed,
            split=split,
            top_k_docs=top_k_docs,
            notes="Improved RAG benchmark.",
        ),
        BenchmarkSpec(
            name="caption",
            model_type="caption",
            checkpoint_path=caption_checkpoint,
            data_csv=data_csv,
            image_dir=image_dir,
            caption_column=caption_column,
            caption_max_length=caption_max_length,
            batch_size=batch_size,
            device=device,
            seed=seed,
            split=split,
            notes="Caption-augmented benchmark.",
        ),
    ]


def run_benchmark_suite(specs: Sequence[BenchmarkSpec], output_dir: Optional[Path] = None):
    runner = BenchmarkRunner(specs, output_dir=output_dir)
    results = runner.run()

    if output_dir is not None:
        runner.save_report(results, Path(output_dir))

    return results
