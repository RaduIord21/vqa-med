from src.vqa_med.config import config

print("Configuration loaded successfully!")
print(config)
print(f"\nData root: {config.paths.data_root}")
print(f"Vision model: {config.model.vision_model}")