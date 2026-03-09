"""
Create sample data for testing the VQA system.
"""
import json
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random

# Sample medical questions and answers
SAMPLE_QA = [
    {"question": "What organ is shown in the image?", "answer": "lung"},
    {"question": "Is there any abnormality visible?", "answer": "yes"},
    {"question": "What is the imaging modality?", "answer": "x-ray"},
    {"question": "What color is highlighted?", "answer": "white"},
    {"question": "Is this a normal scan?", "answer": "no"},
]

def create_sample_images(output_dir: Path, num_samples: int = 20):
    """Create dummy medical images for testing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data = []
    
    for i in range(num_samples):
        # Create a simple colored image
        img = Image.new('RGB', (224, 224), color=(random.randint(0, 255), 
                                                    random.randint(0, 255), 
                                                    random.randint(0, 255)))
        draw = ImageDraw.Draw(img)
        draw.text((50, 100), f"Sample {i}", fill=(255, 255, 255))
        
        # Save image
        img_name = f"sample_{i:03d}.jpg"
        img.save(output_dir / img_name)
        
        # Assign random Q&A
        qa = random.choice(SAMPLE_QA)
        data.append({
            'image_path': img_name,
            'question': qa['question'],
            'answer': qa['answer']
        })
    
    return data

if __name__ == "__main__":
    from vqa_med.config import config
    
    print("Creating sample data...")
    
    # Create sample images
    data = create_sample_images(config.paths.sample_data / "images", num_samples=50)
    
    # Save as CSV
    df = pd.DataFrame(data)
    csv_path = config.paths.sample_data / "sample_vqa.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"Created {len(data)} samples")
    print(f"Images saved to: {config.paths.sample_data / 'images'}")
    print(f"CSV saved to: {csv_path}")
    print(f"\nSample data:")
    print(df.head())