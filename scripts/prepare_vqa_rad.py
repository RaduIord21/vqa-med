"""
Prepare VQA-RAD dataset for training.

Download VQA-RAD from: https://osf.io/89kps/
Extract to data/raw/VQA-RAD/
"""
from pathlib import Path
from vqa_med.utils import prepare_vqa_rad_for_training, get_answer_statistics
from vqa_med.config import config


def main():
    # Paths
    vqa_rad_dir = config.paths.raw_data / "VQA-RAD"
    output_csv = config.paths.processed_data / "vqa_rad_full.csv"
    
    # Check if VQA-RAD exists
    if not vqa_rad_dir.exists():
        print(f"ERROR: VQA-RAD directory not found at {vqa_rad_dir}")
        print("\nPlease download VQA-RAD dataset:")
        print("1. Go to: https://osf.io/89kps/")
        print("2. Download and extract to: data/raw/VQA-RAD/")
        print("\nExpected structure:")
        print("  data/raw/VQA-RAD/")
        print("    ├── images/")
        print("    └── VQA_RAD Dataset Public.json")
        return
    
    print("=" * 60)
    print("Preparing VQA-RAD Dataset")
    print("=" * 60)
    
    # Process full dataset
    df = prepare_vqa_rad_for_training(vqa_rad_dir, output_csv)
    
    # Show statistics
    get_answer_statistics(df)
    
    # Also prepare closed-ended subset (easier to start with)
    closed_csv = config.paths.processed_data / "vqa_rad_closed.csv"
    print("\n" + "=" * 60)
    print("Preparing CLOSED-ENDED subset")
    print("=" * 60)
    df_closed = prepare_vqa_rad_for_training(
        vqa_rad_dir, 
        closed_csv,
        filter_answer_type='CLOSED'
    )
    get_answer_statistics(df_closed)
    
    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  Full dataset: {output_csv}")
    print(f"  Closed-ended: {closed_csv}")


if __name__ == "__main__":
    main()

'''

**Instructions:**

1. **Download VQA-RAD:**
   - Go to: https://osf.io/89kps/
   - Download the dataset
   - Extract to `data/raw/VQA-RAD/`

2. **Expected structure:**
```
   data/raw/VQA-RAD/
   ├── images/
   │   ├── synpic12345.jpg
   │   └── ...
   └── VQA_RAD Dataset Public.json

   '''