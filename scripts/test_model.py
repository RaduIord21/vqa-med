"""
Test the base VQA model architecture.
"""
import torch
from vqa_med.models import BaseVQAModel, VQAModelWrapper
from vqa_med.config import config


def test_model():
    """Test model initialization and forward pass."""
    print("=" * 60)
    print("Testing Base VQA Model")
    print("=" * 60)
    
    # Determine device
    device = torch.device(config.model.device if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Model parameters
    num_classes = 50  # Example: 50 unique answers
    batch_size = 4
    
    # Initialize model
    model = BaseVQAModel(
        num_classes=num_classes,
        hidden_dim=config.model.hidden_dim,
        dropout=config.model.dropout,
    )
    
    # Wrap model
    wrapper = VQAModelWrapper(model, device=str(device))
    
    # Count parameters
    param_counts = wrapper.count_parameters()
    print("\n" + "=" * 60)
    print("Parameter Counts:")
    print("=" * 60)
    for name, count in param_counts.items():
        print(f"  {name.capitalize()}: {count:,}")
    
    # Create dummy inputs ON THE CORRECT DEVICE
    print("\n" + "=" * 60)
    print("Testing forward pass...")
    print("=" * 60)
    
    dummy_image = torch.randn(batch_size, 3, 224, 224).to(device)
    dummy_input_ids = torch.randint(0, 30000, (batch_size, 128)).to(device)
    dummy_attention_mask = torch.ones(batch_size, 128).to(device)
    
    # Forward pass
    try:
        logits = model(dummy_image, dummy_input_ids, dummy_attention_mask)
        print(f"✓ Forward pass successful!")
        print(f"  Input image shape: {dummy_image.shape}")
        print(f"  Input IDs shape: {dummy_input_ids.shape}")
        print(f"  Output logits shape: {logits.shape}")
        print(f"  Expected shape: [{batch_size}, {num_classes}]")
        
        # Test prediction
        predictions, probabilities = wrapper.predict(
            dummy_image, dummy_input_ids, dummy_attention_mask
        )
        print(f"\n✓ Prediction successful!")
        print(f"  Predictions shape: {predictions.shape}")
        print(f"  Probabilities shape: {probabilities.shape}")
        print(f"  Sample prediction: {predictions[0].item()}")
        
    except Exception as e:
        print(f"✗ Error during forward pass: {e}")
        raise
    
    print("\n" + "=" * 60)
    print("Model test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_model()