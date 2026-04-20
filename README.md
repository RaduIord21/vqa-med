# Cell 1: Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Extract dataset (if needed)
!unzip /content/drive/MyDrive/vqa-med-data/archive.zip -d /content/drive/MyDrive/vqa-med-data/

# Cell 3: Organize dataset
!mkdir -p /content/drive/MyDrive/vqa-med-data/VQA-RAD/images
!mv "/content/drive/MyDrive/vqa-med-data/VQA_RAD Image Folder"/* /content/drive/MyDrive/vqa-med-data/VQA-RAD/images/
!mv "/content/drive/MyDrive/vqa-med-data/VQA_RAD Dataset Public.json" /content/drive/MyDrive/vqa-med-data/VQA-RAD/

# Cell 4: Clone repo
!git clone https://github.com/YOUR_USERNAME/vqa-med.git
%cd vqa-med

# Cell 5: Install UV
!curl -LsSf https://astral.sh/uv/install.sh | sh
import os
os.environ['PATH'] = f"{os.path.expanduser('~/.cargo/bin')}:{os.environ['PATH']}"

# Cell 6: Install dependencies
!uv pip install -e .

# Cell 7: Create symlink
!ln -s /content/drive/MyDrive/vqa-med-data/VQA-RAD /content/vqa-med/data/raw/VQA-RAD

# Cell 8: Prepare data
!uv run python scripts/prepare_vqa_rad.py

# Cell 9: Test model
!uv run python scripts/test_model.py

# Cell 10: Train
!mkdir -p /content/drive/MyDrive/vqa-med-checkpoints
!uv run python scripts/train.py \
  --batch_size 16 --num_epochs 10 \
  --checkpoint_dir /content/drive/MyDrive/vqa-med-checkpoints \
  --device cuda

# Cell 11: Train caption-augmented model
!mkdir -p /content/drive/MyDrive/vqa-med-checkpoints/caption
!uv run python scripts/train_caption.py \
  --data_csv /content/drive/MyDrive/vqa-med-data/vqa_rad_closed_with_captions.csv \
  --image_dir data/raw/VQA-RAD/images \
  --caption_column caption \
  --caption_max_length 48 \
  --batch_size 12 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-4 \
  --num_epochs 40 \
  --checkpoint_dir /content/drive/MyDrive/vqa-med-checkpoints/caption \
  --device cuda

# Cell 12: Evaluate caption checkpoint
!uv run python scripts/evaluate_model.py \
  --checkpoint /content/drive/MyDrive/vqa-med-checkpoints/caption/checkpoint_best.pth \
  --data_csv /content/drive/MyDrive/vqa-med-data/vqa_rad_closed_with_captions.csv \
  --output_dir /content/drive/MyDrive/vqa-med-eval/caption \
  --model_type auto \
  --caption_column caption \
  --device cuda