"""
Inference script for Medical VQA model.
"""
import torch
from PIL import Image
from pathlib import Path
import argparse
import pandas as pd

from vqa_med.models import BaseVQAModel, VQAModelWrapper
from vqa_med.data import MedicalVQADataset
from vqa_med.utils import get_image_transforms, get_tokenizer
from vqa_med.config import config


def load_model(checkpoint_path: Path, num_classes: int, device: str = "cuda"):
    """Load trained model from checkpoint."""
    model = BaseVQAModel(
        num_classes=num_classes,
        vision_model_name=config.model.vision_model,
        text_model_name=config.model.text_model,
        hidden_dim=config.model.hidden_dim,
        dropout=config.model.dropout,
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    wrapper = VQAModelWrapper(model, device=device)
    
    print(f"Model loaded from: {checkpoint_path}")
    print(f"Trained for {checkpoint['epoch']} epochs")
    print(f"Validation accuracy: {checkpoint['val_acc']:.2f}%")
    
    return wrapper


def predict_single(
    model_wrapper: VQAModelWrapper,
    image_path: Path,
    question: str,
    transform,
    tokenizer,
    idx_to_answer: dict,
    top_k: int = 5,
):
    """Make prediction on a single image-question pair."""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Tokenize question
    question_encoded = tokenizer(
        question,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Predict
    pred_idx, probs = model_wrapper.predict(
        image_tensor,
        question_encoded['input_ids'],
        question_encoded['attention_mask']
    )
    
    # Get answer
    predicted_answer = idx_to_answer[pred_idx.item()]
    confidence = probs[0, pred_idx].item()
    
    # Get top K predictions
    topk_probs, topk_indices = torch.topk(probs[0], k=min(top_k, len(idx_to_answer)))
    topk_answers = [(idx_to_answer[idx.item()], prob.item()) 
                    for idx, prob in zip(topk_indices, topk_probs)]
    
    return predicted_answer, confidence, topk_answers


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Medical VQA Inference')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (default: best checkpoint)')
    parser.add_argument('--data_csv', type=str, default=None,
                        help='Path to CSV file with QA pairs (for answer vocabulary)')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Directory containing images')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cuda', 'cpu'],
                        help='Device to run inference on')
    
    # Inference mode
    parser.add_argument('--mode', type=str, default='interactive',
                        choices=['interactive', 'single', 'batch'],
                        help='Inference mode')
    
    # Single prediction arguments
    parser.add_argument('--image', type=str, default=None,
                        help='Image path for single prediction')
    parser.add_argument('--question', type=str, default=None,
                        help='Question for single prediction')
    
    # Batch prediction arguments
    parser.add_argument('--batch_csv', type=str, default=None,
                        help='CSV file with image_path and question columns')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Output CSV file for batch predictions')
    
    # Other arguments
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top predictions to show')
    
    return parser.parse_args()


def interactive_mode(model, transform, tokenizer, idx_to_answer, image_dir, top_k):
    """Interactive inference mode."""
    print("\n" + "=" * 60)
    print("Interactive Mode")
    print("Enter image path and question (or 'quit' to exit)")
    print("=" * 60)
    
    while True:
        print("\n")
        image_path_str = input("Image path (relative to image dir): ").strip()
        
        if image_path_str.lower() == 'quit':
            break
        
        question = input("Question: ").strip()
        
        if not question:
            continue
        
        full_image_path = image_dir / image_path_str
        
        if not full_image_path.exists():
            print(f"ERROR: Image not found at {full_image_path}")
            continue
        
        # Predict
        answer, confidence, topk = predict_single(
            model,
            full_image_path,
            question,
            transform,
            tokenizer,
            idx_to_answer,
            top_k,
        )
        
        print("\n" + "-" * 60)
        print(f"Predicted Answer: {answer}")
        print(f"Confidence: {confidence:.2%}")
        print(f"\nTop {top_k} Predictions:")
        for i, (ans, prob) in enumerate(topk, 1):
            print(f"  {i}. {ans}: {prob:.2%}")
        print("-" * 60)


def single_mode(model, transform, tokenizer, idx_to_answer, image_path, question, top_k):
    """Single prediction mode."""
    print("\n" + "=" * 60)
    print("Single Prediction Mode")
    print("=" * 60)
    
    if not Path(image_path).exists():
        print(f"ERROR: Image not found at {image_path}")
        return
    
    print(f"\nImage: {image_path}")
    print(f"Question: {question}")
    
    # Predict
    answer, confidence, topk = predict_single(
        model,
        Path(image_path),
        question,
        transform,
        tokenizer,
        idx_to_answer,
        top_k,
    )
    
    print("\n" + "-" * 60)
    print(f"Predicted Answer: {answer}")
    print(f"Confidence: {confidence:.2%}")
    print(f"\nTop {top_k} Predictions:")
    for i, (ans, prob) in enumerate(topk, 1):
        print(f"  {i}. {ans}: {prob:.2%}")
    print("-" * 60)


def batch_mode(model, transform, tokenizer, idx_to_answer, batch_csv, output_csv, image_dir, top_k):
    """Batch prediction mode."""
    print("\n" + "=" * 60)
    print("Batch Prediction Mode")
    print("=" * 60)
    
    # Load batch data
    df = pd.read_csv(batch_csv)
    
    if 'image_path' not in df.columns or 'question' not in df.columns:
        print("ERROR: CSV must contain 'image_path' and 'question' columns")
        return
    
    print(f"\nProcessing {len(df)} samples...")
    
    results = []
    
    from tqdm import tqdm
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_path = image_dir / row['image_path']
        question = row['question']
        
        if not image_path.exists():
            print(f"WARNING: Image not found: {image_path}")
            results.append({
                'image_path': row['image_path'],
                'question': question,
                'predicted_answer': 'N/A',
                'confidence': 0.0,
            })
            continue
        
        # Predict
        answer, confidence, topk = predict_single(
            model,
            image_path,
            question,
            transform,
            tokenizer,
            idx_to_answer,
            top_k,
        )
        
        results.append({
            'image_path': row['image_path'],
            'question': question,
            'predicted_answer': answer,
            'confidence': confidence,
            **{f'top_{i}_answer': ans for i, (ans, _) in enumerate(topk, 1)},
            **{f'top_{i}_prob': prob for i, (_, prob) in enumerate(topk, 1)},
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    
    print(f"\n✓ Results saved to: {output_csv}")
    print(f"Total predictions: {len(results_df)}")


def main():
    """Main inference function."""
    args = parse_args()
    
    print("=" * 60)
    print("Medical VQA Inference")
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
    
    if not data_csv.exists():
        print(f"ERROR: Data file not found at {data_csv}")
        return
    
    # Load dataset (just to get answer vocabulary)
    transform = get_image_transforms(config.model.image_size, is_training=False)
    tokenizer = get_tokenizer(config.model.text_model)
    
    dataset = MedicalVQADataset(
        data_file=data_csv,
        image_dir=image_dir,
        transform=transform,
        tokenizer=tokenizer,
    )
    
    idx_to_answer = dataset.idx_to_answer
    num_classes = len(idx_to_answer)
    
    # Load model
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = config.paths.checkpoints_dir / "checkpoint_best.pth"
    
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first: uv run python scripts/train.py")
        return
    
    device = args.device or config.model.device
    model = load_model(checkpoint_path, num_classes, device)
    
    # Run appropriate mode
    if args.mode == 'interactive':
        interactive_mode(model, transform, tokenizer, idx_to_answer, image_dir, args.top_k)
    
    elif args.mode == 'single':
        if not args.image or not args.question:
            print("ERROR: --image and --question required for single mode")
            return
        single_mode(model, transform, tokenizer, idx_to_answer, 
                   args.image, args.question, args.top_k)
    
    elif args.mode == 'batch':
        if not args.batch_csv or not args.output_csv:
            print("ERROR: --batch_csv and --output_csv required for batch mode")
            return
        batch_mode(model, transform, tokenizer, idx_to_answer, 
                  args.batch_csv, args.output_csv, image_dir, args.top_k)


if __name__ == "__main__":
    main()