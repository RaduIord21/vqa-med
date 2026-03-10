"""
Comprehensive model evaluation script.
Analyzes performance by question type, answer distribution, and failure cases.
"""
import os
os.environ['MPLBACKEND'] = 'Agg'

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse
from tqdm import tqdm
from collections import defaultdict

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score

from vqa_med.models import BaseVQAModel, VQAModelWrapper
from vqa_med.data import MedicalVQADataset
from vqa_med.utils import get_image_transforms, get_tokenizer
from vqa_med.config import config


class VQAEvaluator:
    """Comprehensive VQA model evaluator."""
    
    def __init__(
        self,
        model_wrapper: VQAModelWrapper,
        dataset: MedicalVQADataset,
        device: str = "cuda"
    ):
        self.model = model_wrapper
        self.dataset = dataset
        self.device = device
        self.idx_to_answer = dataset.idx_to_answer
        self.answer_to_idx = dataset.answer_to_idx
        
        # Results storage
        self.predictions = []
        self.ground_truths = []
        self.confidences = []
        self.question_types = []
        self.answer_types = []
        self.questions = []
        self.images = []
        
    def evaluate(self, batch_size: int = 16):
        """Run evaluation on entire dataset."""
        print("Running evaluation...")
        
        from torch.utils.data import DataLoader
        loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
        )
        
        self.model.model.eval()
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating"):
                # Get predictions
                images = batch['image'].to(self.device)
                input_ids = batch['question']['input_ids'].to(self.device)
                attention_mask = batch['question']['attention_mask'].to(self.device)
                
                pred_indices, probs = self.model.predict(images, input_ids, attention_mask)
                
                # Store results
                for i in range(len(pred_indices)):
                    pred_idx = pred_indices[i].item()
                    true_idx = batch['answer'][i].item()
                    confidence = probs[i, pred_idx].item()
                    
                    self.predictions.append(self.idx_to_answer[pred_idx])
                    self.ground_truths.append(batch['answer_text'][i])
                    self.confidences.append(confidence)
                    self.questions.append(batch['question_text'][i])
                    self.images.append(batch['image_path'][i] if 'image_path' in batch else 'N/A')
                    
                    # Get question type and answer type from dataset
                    idx = len(self.predictions) - 1
                    data_row = self.dataset.data.iloc[idx]
                    self.question_types.append(data_row.get('question_type', 'unknown'))
                    self.answer_types.append(data_row.get('answer_type', 'unknown'))
        
        print(f"Evaluation complete: {len(self.predictions)} samples")
    
    def calculate_metrics(self):
        """Calculate overall and per-type metrics."""
        # Overall accuracy
        correct = sum([p == g for p, g in zip(self.predictions, self.ground_truths)])
        overall_acc = 100.0 * correct / len(self.predictions)
        
        # Per question-type accuracy
        qt_metrics = defaultdict(lambda: {'correct': 0, 'total': 0})
        for pred, gt, qt in zip(self.predictions, self.ground_truths, self.question_types):
            qt_metrics[qt]['total'] += 1
            if pred == gt:
                qt_metrics[qt]['correct'] += 1
        
        qt_accuracy = {
            qt: 100.0 * metrics['correct'] / metrics['total']
            for qt, metrics in qt_metrics.items()
        }
        
        # Per answer-type accuracy
        at_metrics = defaultdict(lambda: {'correct': 0, 'total': 0})
        for pred, gt, at in zip(self.predictions, self.ground_truths, self.answer_types):
            at_metrics[at]['total'] += 1
            if pred == gt:
                at_metrics[at]['correct'] += 1
        
        at_accuracy = {
            at: 100.0 * metrics['correct'] / metrics['total']
            for at, metrics in at_metrics.items()
        }
        
        # Average confidence
        avg_confidence = np.mean(self.confidences)
        
        # Confidence for correct vs incorrect
        correct_mask = [p == g for p, g in zip(self.predictions, self.ground_truths)]
        correct_confidences = [c for c, m in zip(self.confidences, correct_mask) if m]
        incorrect_confidences = [c for c, m in zip(self.confidences, correct_mask) if not m]
        
        metrics = {
            'overall_accuracy': overall_acc,
            'question_type_accuracy': qt_accuracy,
            'answer_type_accuracy': at_accuracy,
            'avg_confidence': avg_confidence,
            'correct_avg_confidence': np.mean(correct_confidences) if correct_confidences else 0,
            'incorrect_avg_confidence': np.mean(incorrect_confidences) if incorrect_confidences else 0,
            'total_samples': len(self.predictions),
            'question_type_distribution': dict(qt_metrics),
            'answer_type_distribution': dict(at_metrics),
        }
        
        return metrics
    
    def get_confusion_matrix(self, top_n: int = 20):
        """Get confusion matrix for top N most frequent answers."""
        # Get top N answers
        from collections import Counter
        answer_counts = Counter(self.ground_truths)
        top_answers = [ans for ans, _ in answer_counts.most_common(top_n)]
        
        # Filter to only top answers
        filtered_preds = []
        filtered_gts = []
        
        for pred, gt in zip(self.predictions, self.ground_truths):
            if gt in top_answers:
                filtered_preds.append(pred if pred in top_answers else 'other')
                filtered_gts.append(gt)
        
        # Create confusion matrix
        labels = top_answers + ['other']
        cm = confusion_matrix(filtered_gts, filtered_preds, labels=labels)
        
        return cm, labels
    
    def get_failure_cases(self, n: int = 50):
        """Get worst failure cases (lowest confidence incorrect predictions)."""
        failures = []
        
        for i, (pred, gt, conf, qt, q, img) in enumerate(zip(
            self.predictions, self.ground_truths, self.confidences,
            self.question_types, self.questions, self.images
        )):
            if pred != gt:
                failures.append({
                    'index': i,
                    'image': img,
                    'question': q,
                    'question_type': qt,
                    'predicted': pred,
                    'ground_truth': gt,
                    'confidence': conf,
                })
        
        # Sort by confidence (high confidence wrong predictions are worst)
        failures.sort(key=lambda x: x['confidence'], reverse=True)
        
        return failures[:n]
    
    def get_answer_distribution(self):
        """Analyze answer distribution."""
        from collections import Counter
        
        gt_dist = Counter(self.ground_truths)
        pred_dist = Counter(self.predictions)
        
        return {
            'ground_truth': dict(gt_dist),
            'predictions': dict(pred_dist),
        }
    
    def save_results(self, output_dir: Path):
        """Save evaluation results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        
        # Save metrics as JSON
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save detailed results as CSV
        results_df = pd.DataFrame({
            'image': self.images,
            'question': self.questions,
            'question_type': self.question_types,
            'answer_type': self.answer_types,
            'predicted': self.predictions,
            'ground_truth': self.ground_truths,
            'confidence': self.confidences,
            'correct': [p == g for p, g in zip(self.predictions, self.ground_truths)],
        })
        results_df.to_csv(output_dir / 'detailed_results.csv', index=False)
        
        # Save failure cases
        failures = self.get_failure_cases(100)
        failures_df = pd.DataFrame(failures)
        failures_df.to_csv(output_dir / 'failure_cases.csv', index=False)
        
        # Save answer distribution
        answer_dist = self.get_answer_distribution()
        with open(output_dir / 'answer_distribution.json', 'w') as f:
            json.dump(answer_dist, f, indent=2)
        
        print(f"\nResults saved to: {output_dir}")
        
        return metrics
    
    def plot_results(self, output_dir: Path):
        """Generate visualizations."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Question type accuracy
        metrics = self.calculate_metrics()
        qt_acc = metrics['question_type_accuracy']
        
        plt.figure(figsize=(12, 6))
        plt.bar(qt_acc.keys(), qt_acc.values())
        plt.xlabel('Question Type')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy by Question Type')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / 'question_type_accuracy.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Answer type accuracy
        at_acc = metrics['answer_type_accuracy']
        
        plt.figure(figsize=(10, 6))
        plt.bar(at_acc.keys(), at_acc.values())
        plt.xlabel('Answer Type')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy by Answer Type')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / 'answer_type_accuracy.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Confusion matrix
        cm, labels = self.get_confusion_matrix(top_n=15)
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('Ground Truth')
        plt.title('Confusion Matrix (Top 15 Answers)')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 4. Confidence distribution
        correct_mask = [p == g for p, g in zip(self.predictions, self.ground_truths)]
        correct_conf = [c for c, m in zip(self.confidences, correct_mask) if m]
        incorrect_conf = [c for c, m in zip(self.confidences, correct_mask) if not m]
        
        plt.figure(figsize=(10, 6))
        plt.hist(correct_conf, bins=50, alpha=0.5, label='Correct', color='green')
        plt.hist(incorrect_conf, bins=50, alpha=0.5, label='Incorrect', color='red')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.title('Confidence Distribution: Correct vs Incorrect Predictions')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'confidence_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 5. Answer frequency comparison
        answer_dist = self.get_answer_distribution()
        gt_dist = answer_dist['ground_truth']
        pred_dist = answer_dist['predictions']
        
        # Top 20 answers
        from collections import Counter
        top_answers = [ans for ans, _ in Counter(gt_dist).most_common(20)]
        
        gt_counts = [gt_dist.get(ans, 0) for ans in top_answers]
        pred_counts = [pred_dist.get(ans, 0) for ans in top_answers]
        
        x = np.arange(len(top_answers))
        width = 0.35
        
        plt.figure(figsize=(14, 6))
        plt.bar(x - width/2, gt_counts, width, label='Ground Truth', alpha=0.8)
        plt.bar(x + width/2, pred_counts, width, label='Predictions', alpha=0.8)
        plt.xlabel('Answer')
        plt.ylabel('Frequency')
        plt.title('Answer Distribution: Ground Truth vs Predictions (Top 20)')
        plt.xticks(x, top_answers, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'answer_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to: {output_dir}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate Medical VQA Model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_csv', type=str, default=None,
                        help='Path to evaluation data CSV')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Directory containing images')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to run evaluation on')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test', 'all'],
                        help='Which split to evaluate on')
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    print("=" * 60)
    print("Medical VQA Model Evaluation")
    print("=" * 60)
    
    # Data paths
    if args.data_csv:
        data_csv = Path(args.data_csv)
    else:
        data_csv = config.paths.processed_data / "vqa_rad_closed.csv"
    
    if args.image_dir:
        image_dir = Path(args.image_dir)
    else:
        image_dir = config.paths.raw_data / "VQA-RAD" / "images"
    
    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = config.paths.outputs_dir / "evaluation"
    
    # Load dataset
    print("\nLoading dataset...")
    transform = get_image_transforms(config.model.image_size, is_training=False)
    tokenizer = get_tokenizer(config.model.text_model)
    
    full_dataset = MedicalVQADataset(
        data_file=data_csv,
        image_dir=image_dir,
        transform=transform,
        tokenizer=tokenizer,
    )
    
    # Get appropriate split
    if args.split != 'all':
        from torch.utils.data import random_split
        total_size = len(full_dataset)
        train_size = int(total_size * 0.7)
        val_size = int(total_size * 0.15)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        if args.split == 'train':
            eval_dataset = train_dataset
        elif args.split == 'val':
            eval_dataset = val_dataset
        else:  # test
            eval_dataset = test_dataset
        
        print(f"Evaluating on {args.split} split: {len(eval_dataset)} samples")
    else:
        eval_dataset = full_dataset
        print(f"Evaluating on full dataset: {len(eval_dataset)} samples")
    
    # Load model
    print("\nLoading model...")
    checkpoint_path = Path(args.checkpoint)
    
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return
    
    num_classes = full_dataset.get_num_classes()
    
    model = BaseVQAModel(
        num_classes=num_classes,
        vision_model_name=config.model.vision_model,
        text_model_name=config.model.text_model,
        hidden_dim=config.model.hidden_dim,
        dropout=config.model.dropout,
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model_wrapper = VQAModelWrapper(model, device=args.device)
    
    print(f"Model loaded from: {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Checkpoint val acc: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    
    # Create evaluator
    evaluator = VQAEvaluator(model_wrapper, eval_dataset, device=args.device)
    
    # Run evaluation
    evaluator.evaluate(batch_size=args.batch_size)
    
    # Calculate and save metrics
    metrics = evaluator.save_results(output_dir)
    
    # Generate plots
    evaluator.plot_results(output_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.2f}%")
    print(f"Average Confidence: {metrics['avg_confidence']:.2%}")
    print(f"Correct Predictions Confidence: {metrics['correct_avg_confidence']:.2%}")
    print(f"Incorrect Predictions Confidence: {metrics['incorrect_avg_confidence']:.2%}")
    
    print("\n" + "-" * 60)
    print("Accuracy by Question Type:")
    print("-" * 60)
    for qt, acc in sorted(metrics['question_type_accuracy'].items(), key=lambda x: x[1], reverse=True):
        count = metrics['question_type_distribution'][qt]['total']
        print(f"  {qt:20s}: {acc:6.2f}% ({count:4d} samples)")
    
    print("\n" + "-" * 60)
    print("Accuracy by Answer Type:")
    print("-" * 60)
    for at, acc in sorted(metrics['answer_type_accuracy'].items(), key=lambda x: x[1], reverse=True):
        count = metrics['answer_type_distribution'][at]['total']
        print(f"  {at:20s}: {acc:6.2f}% ({count:4d} samples)")
    
    print("\n" + "=" * 60)
    print(f"Detailed results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()