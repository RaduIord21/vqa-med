# Medical VQA Project - Development Session Summary

**Date:** March 31 - April 2, 2026 (updated April 2 baseline lock)  
**Project:** Medical Visual Question Answering System  
**Developer:** Radu  
**Environment:** Google Colab Pro+ (A100 80GB GPU)

### **April 2 Baseline Lock Update**
- Caption-augmented model reached **69.59%** validation accuracy using cached captions, `--learning_rate 1e-4`, and `--caption_max_length 48`.
- This currently exceeds the previous attention baseline (**69.07%**) and is the working baseline for adversarial prompting experiments.
- Next decision criterion: verify reproducibility around 69.59% before large robustness training sweeps.

---

## **Project Overview**

Building a Medical Visual Question Answering (VQA) system that can answer questions about medical images (X-rays, CT, MRI scans). The project uses the VQA-RAD dataset from Kaggle and progressively enhances a base model with advanced techniques (RAG, attention mechanisms, image captioning, adversarial prompting).

---

## **Technology Stack**

- **Package Manager:** UV
- **Deep Learning:** PyTorch, Transformers (HuggingFace)
- **Vision Model:** ViT-base-patch16-224 (86M params)
- **Text Model:** BiomedNLP-PubMedBERT-base-uncased-abstract (109M params)
- **Dataset:** VQA-RAD (1,297 samples, 64 unique answers)
- **Training Hardware:** Google Colab Pro+ A100 80GB GPU
- **Storage:** Google Drive for data and checkpoints

---

## **Project Structure**

```
vqa-med/
├── pyproject.toml                 # UV project configuration (Hatchling)
├── README.md
├── TODO.md                        # Progress tracking
├── data/
│   ├── raw/VQA-RAD/              # Dataset (from Kaggle)
│   │   ├── images/
│   │   └── VQA_RAD Dataset Public.json
│   ├── processed/                 # Processed CSV files
│   └── knowledge/                 # RAG knowledge base
├── src/vqa_med/                  # Main package
│   ├── models/
│   │   ├── base_vqa.py           # Simple concatenation model
│   │   ├── attention_vqa.py      # Cross-attention model ✅ BEST
│   │   ├── caption_vqa.py        # Caption-augmented VQA model (new)
│   │   ├── rag_vqa.py            # RAG-enhanced (underperforming)
│   │   └── rag_vqa_improved.py   # Improved RAG (not tested)
│   ├── data/
│   │   └── dataset.py            # Dataset loader
│   ├── retrieval/                # RAG components
│   │   ├── knowledge_base.py     # FAISS vector database
│   │   └── retriever.py          # Medical knowledge retriever
│   ├── utils/
│   │   └── helpers.py            # Transforms, tokenizers, metrics
│   └── config/
│       └── settings.py           # Configuration
├── scripts/
│   ├── prepare_vqa_rad.py        # Dataset preprocessing
│   ├── train.py                  # Base training
│   ├── train_balanced.py         # Balanced sampling training
│   ├── train_specialized.py      # Question-type specific models
│   ├── train_attention.py        # Attention model training
│   ├── train_attention_fast.py   # Optimized training (FP16)
│   ├── train_caption.py          # Caption-augmented training (new)
│   ├── train_rag.py              # RAG training
│   ├── evaluate_model.py         # Comprehensive evaluation
│   ├── analyze_dataset.py        # Dataset analysis
│   └── build_knowledge_base.py   # Create RAG knowledge base
└── notebooks/
    └── error_analysis.ipynb      # Interactive error analysis
```

---

## **Dataset Characteristics (VQA-RAD Closed-Ended)**

**Critical Findings:**
- **Total samples:** 1,297
- **Unique images:** 315
- **Unique answers:** 64
- **Severe class imbalance:** 92% of answers are "yes" or "no"
  - "no": 606 samples (46.7%)
  - "yes": 586 samples (45.2%)
  - All other answers: 105 samples (8.1%)
- **Imbalance ratio:** 606:1 (most common to least common)
- **Gini coefficient:** 0.910 (extreme inequality)

**Question Type Distribution:**
- PRES (Presence): 47.6% - "Is there X?"
- SIZE: 12.0%
- ABN (Abnormality): 9.6%
- MODALITY: 7.1%
- PLANE: 4.5%
- ORGAN: 1.3%
- POS (Position): 3.5% - **Completely failed (0% accuracy)**

---

## **Development Progress & Results**

### **Phase 1: Base Model Development**

#### **Step 1-6: Project Setup & Base Model**
- ✅ Created project structure with UV + Hatchling
- ✅ Implemented data loading pipeline for VQA-RAD
- ✅ Built base VQA model (ViT + BioBERT + simple concatenation)
- ✅ Created training and inference scripts with CLI arguments
- ✅ Setup for Google Colab Pro+ compatibility

**Base Model Architecture:**
- Vision: ViT-base extracts image features
- Text: BioBERT extracts question features
- Fusion: Simple concatenation of [CLS] tokens
- Classifier: MLP → Answer prediction

**Result:** 55.61% validation accuracy (barely better than random for yes/no)

---

### **Phase 2: Analysis & Understanding the Problem**

#### **Step 7: Dataset & Model Analysis**
Created comprehensive analysis tools:
- ✅ `scripts/analyze_dataset.py` - Dataset statistics and distributions
- ✅ `scripts/evaluate_model.py` - Per-question-type performance analysis
- ✅ `notebooks/error_analysis.ipynb` - Interactive error exploration

**Key Findings:**
1. **Class imbalance is severe** but not the root cause
2. **Model isn't learning visual features properly**
3. **Simple concatenation fusion is too weak**
4. **Even yes/no questions only get 61% accuracy** (should be 85%+)

**Performance by Question Type (Base Model):**
- PRES (yes/no): 59.14% - Should be much higher
- MODALITY: 70.00% - Best performer
- ABN: 59.09%
- POS: 0.00% - **Complete failure**
- Overall: 55.61%

---

### **Phase 3: Attempted Fixes (Path A)**

#### **Attempt 1: Balanced Sampling**
**Script:** `train_balanced.py`  
**Approach:** Weighted sampling to balance rare answers  
**Result:** 56.70% accuracy (no improvement)  
**Conclusion:** Class imbalance is not the root problem

#### **Attempt 2: Specialized Models**
**Script:** `train_specialized.py`  
**Approach:** Train separate models for different question types

**Results:**
- Binary model (yes/no only): 61.24% - Still poor!
- Specific answers (non-yes/no): 13.33% - Failed completely
- Modality model: 63.64%

**Conclusion:** Even isolated yes/no questions fail. Problem is deeper.

---

### **Phase 4: Architecture Improvement (Breakthrough!)**

#### **Cross-Attention Model** ✅ **BEST SOLUTION**
**Script:** `train_attention.py` (also `train_attention_fast.py` with FP16)  
**Architecture Changes:**
1. Cross-attention between question and image patches
2. Multi-head attention (8 heads)
3. Better fusion with attended features
4. Deeper MLP layers

**Key Innovation:**
```python
# Question attends to image patches
attended_vision = cross_attention(
    query=text_features,    # Question tokens
    key=vision_features,    # Image patches
    value=vision_features
)
# Then fuse attended vision with text
```

**Result:** 69.07% validation accuracy (+13.5% improvement!) 🎉

**Training Configuration:**
- Batch size: 24-48 (depending on GPU)
- Learning rate: 5e-4 (different rates for pretrained vs new layers)
- Epochs: 40
- Optimizer: AdamW with cosine annealing
- Mixed precision: FP16 enabled (~2x speedup)
- Device: A100 80GB GPU
- Training time: ~25-30 minutes

**Why This Worked:**
- Model now **actually looks at relevant image regions** based on question
- Cross-attention creates visual-linguistic alignment
- Attended features are more informative than global [CLS] token

---

### **Phase 5: RAG Implementation (Implemented + Debugged)**

#### **RAG-Enhanced VQA Model**
**Script:** `train_rag.py`  
**Components Created:**
1. `retrieval/knowledge_base.py` - FAISS vector database
2. `retrieval/retriever.py` - Medical knowledge retriever  
3. `models/rag_vqa.py` - RAG-integrated VQA model
4. `build_knowledge_base.py` - Knowledge base creation

**Knowledge Base:**
- ~35 sample medical documents (radiology facts, anatomy, common findings)
- Embedded with sentence-transformers
- FAISS L2 index for similarity search
- Top-K=3 retrieval per question

**RAG Architecture:**
```
Question → Retrieve Medical Knowledge (Top-3 docs)
                ↓
Image + Question + Context → Cross-Attention → Answer
```

**Result:** 65% validation accuracy (WORSE than 69% baseline!)

**Why RAG Underperformed:**
1. **Knowledge base too small** (only 35 documents)
2. **Generic facts don't match specific questions**
3. **Simple fusion** (averaging embeddings loses information)
4. **Retrieval mismatch** (query doesn't include visual context)

**Papers Provided for Guidance:**
- Multiple papers attached by user (not fully analyzed yet)
- Need to extract better RAG fusion strategies from literature

**Improved RAG Implemented and Tested:**
- `models/rag_vqa_improved.py`
- `scripts/train_rag_improved.py`
- `src/vqa_med/retrieval/retriever.py`

**Major updates completed in this session:**
1. Visual-aware retrieval added and debugged (shape + dimension fixes)
2. Knowledge base expanded (from ~35 to 140+ documents)
3. Multi-query retrieval expansion added (now optional)
4. Learnable temperature scaling added to trainer (now optional)
5. Argument/parser and dataset init issues fixed in `train_rag_improved.py`
6. Accuracy utility hardened for logits vs class-index input

**Recent RAG runs:**
- Tuned/experimental run: **68.46%**
- Aggressive run (query expansion + temperature learning): **66.54%**

**Interpretation:**
- RAG is close to baseline but not consistently better yet
- Aggressive retrieval/training tricks can reduce performance on this small dataset
- Stable ablation is required (enable one enhancement at a time)

---

### **Phase 6: Image Captioning Integration (Implemented + First Run)**

#### **Caption-Augmented VQA Model (April 1, 2026)**
**New components added:**
1. `src/vqa_med/models/caption_vqa.py` - Caption-aware fusion model
2. `scripts/train_caption.py` - End-to-end trainer with optional BLIP caption generation
3. `src/vqa_med/models/__init__.py` updated to export `CaptionVQAModel`

**Architecture Summary:**
1. ViT image encoder
2. Shared BioBERT encoder for question and caption text
3. Cross-attention (question -> image patches)
4. Gated fusion of question CLS + attended vision CLS + caption CLS
5. MLP classifier for answer prediction

**Training Pipeline Features:**
- Optional automatic caption generation (`--generate_captions`) using BLIP
- Caption CSV caching to avoid regenerating captions every run
- Mixed precision (FP16) training support
- Gradient accumulation and cosine LR scheduling

**First Captioning Result:**
- **Best validation accuracy: 66.49%**
- This is **below** the current attention baseline (**69.07%**)

**Current interpretation:**
- Captioning path is integrated and runnable, but first configuration underperformed
- Need controlled caption ablations (learning rate, caption length, cached captions only)

---

## **Current State**

### **Best Model Performance**
- **Best overall model:** Caption-augmented VQA
- **Validation Accuracy:** **69.59%**
- **Run config:** cached captions, `--learning_rate 1e-4`, `--caption_max_length 48`, `--batch_size 12`, `--gradient_accumulation_steps 2`, `--num_epochs 40`
- **Current reference threshold:** 69.07% (attention baseline)

### **Current RAG Status (as of March 31, 2026)**
- **Script:** `scripts/train_rag_improved.py`
- **Most recent scores:** 68.46% and 66.54% (depending on settings)
- **Conclusion:** RAG pipeline is now runnable and debuggable, but still below 69.07% baseline

### **Current Captioning Status (as of April 2, 2026)**
- **Script:** `scripts/train_caption.py`
- **Most recent best score:** **69.59%**
- **Conclusion:** Captioning now beats the attention baseline in at least one controlled run; next step is reproducibility checks before adversarial training.

### **GPU Utilization**
- **Available:** A100 80GB
- **Currently Used:** ~10GB (only 12.5%)
- **Opportunity:** Can increase batch size to 128-192 or use larger models

### **What's Working**
✅ Cross-attention architecture  
✅ Mixed precision training (FP16)  
✅ Comprehensive evaluation tools  
✅ Dataset analysis pipelines  
✅ CLI-based training scripts  
✅ `train_rag_improved.py` end-to-end training now runs without runtime crashes  
✅ Visual reranking no longer fails due to tensor shape/dimension mismatch  
✅ Caption-augmented model + training script implemented and Colab runnable  
✅ Caption ablation run reached 69.59% and exceeded the 69.07% attention baseline  

### **What's Not Working Yet**
❌ RAG still not consistently beating 69.07% baseline  
❌ Position questions (POS) still at 0% accuracy  
❌ Non-yes/no questions underperform  
❌ Retrieval quality still noisy for some question types  
❌ Caption improvement not yet confirmed as stable across repeated runs  

---

## **Next Steps (Tomorrow Plan)**

### **Priority 1: Captioning Ablation (Now Active)**
**Status:** In progress (first run completed)

Run in this order:
1. Cached captions only (no generation during training)
2. Lower LR (`2e-4`)
3. Shorter caption length (`--caption_max_length 32` or 48)
4. Keep same data split/seed for clean comparison

Goal: determine whether captioning can beat 69.07% with stable settings.

### **Priority 2: RAG Decision Gate**
**Status:** Paused unless captioning stalls

If caption ablation does not beat baseline, run one final stable RAG sanity check and then pick the better of RAG vs captioning for next development cycle.

### **Priority 3: Adversarial Prompting**
**Status:** Not started

**Purpose:**
- Test model robustness
- Handle unanswerable questions
- Detect misleading assumptions in questions
- Add "uncertain" / "not applicable" as answer option

**Examples:**
- "What color is the dog?" (when no dog in image)
- "Is the patient male?" (when gender not visible)
- "Where is the tumor?" (when no tumor present)

### **Priority 4: Leverage Unused GPU Memory**
**Current:** 10GB / 80GB used  
**Options:**
1. Increase batch size to 128-192 (better gradients)
2. Use larger models (ViT-Large + BioBERT-Large)
3. Train ensemble of models
4. Higher resolution images (512×512 instead of 224×224)

---

## **Key Learnings**

### **What Worked**
1. **Cross-attention is crucial** - Simple concatenation fails
2. **Mixed precision training** - 2x speedup with minimal accuracy loss
3. **Comprehensive analysis first** - Understanding the problem before solutions
4. **Modular architecture** - Easy to swap components and experiment

### **What Didn't Work**
1. **Class balancing alone** - Wasn't the core issue
2. **Specialized models** - Even yes/no models failed
3. **RAG with small KB** - 35 documents insufficient
4. **Simple fusion strategies** - Averaging/concatenation too weak

### **Critical Insights**
- **92% yes/no imbalance dominates** but isn't the root cause
- **Visual features not learned** without proper attention mechanisms
- **Small medical VQA datasets are challenging** - need smart architectures
- **Position questions need special handling** - spatial reasoning is hard
- **Captioning integration alone is not sufficient** - fusion/training settings matter

---

## **How to Continue This Work**

### **1. Resume Training Session**
```bash
# On Google Colab, clone repo
!git clone https://github.com/YOUR_USERNAME/vqa-med.git
%cd vqa-med

# Install UV
!curl -LsSf https://astral.sh/uv/install.sh | sh
!export PATH="$HOME/.cargo/bin:$PATH"

# Install dependencies
!uv pip install -e .

# Mount Drive for checkpoints
from google.colab import drive
drive.mount('/content/drive')
```

### **2. Continue from Best Model / RAG Ablation**
```bash
# Stable improved RAG run (recommended first run tomorrow)
!uv run python scripts/train_rag_improved.py \
  --data_path data/processed/vqa_rad_closed.csv \
  --image_dir data/raw/VQA-RAD/images \
  --batch_size 12 \
  --gradient_accumulation_steps 2 \
  --learning_rate 5e-4 \
  --num_epochs 40 \
  --checkpoint_dir /content/drive/MyDrive/vqa-checkpoints-rag-stable \
  --device cuda \
  --visual_weight 0.25 \
  --temperature_init 1.0 \
  --top_k_docs 3 \
  --use_gated_fusion

# Optional ablation toggles for later runs:
#   --use_query_expansion
#   --learn_temperature
```

### **3. Evaluate Any Model**
```bash
!uv run python scripts/evaluate_model.py \
  --checkpoint /content/drive/MyDrive/vqa-checkpoints-attention-fast/checkpoint_best.pth \
  --split test \
  --output_dir /content/drive/MyDrive/vqa-evaluation
```

---

## **Important Files & Locations**

### **Data**
- **VQA-RAD Dataset:** `/content/drive/MyDrive/vqa-med-data/VQA-RAD/`
- **Processed CSV:** `data/processed/vqa_rad_closed.csv`
- **Knowledge Base:** `data/knowledge/medical_kb/`

### **Checkpoints**
- **Best Model:** `/content/drive/MyDrive/vqa-checkpoints-attention-fast/checkpoint_best.pth`
- **All Models:** Various checkpoint directories in Drive

### **Key Scripts**
- **Training:** `scripts/train_attention_fast.py` (current best)
- **Caption training:** `scripts/train_caption.py` (new)
- **Evaluation:** `scripts/evaluate_model.py`
- **Analysis:** `scripts/analyze_dataset.py`
- **RAG baseline:** `scripts/train_rag.py`
- **RAG improved:** `scripts/train_rag_improved.py` (active experiment script)

---

## **Workflow Summary**

### **Typical Training Workflow**
1. Prepare/verify data: `prepare_vqa_rad.py`
2. Analyze dataset: `analyze_dataset.py`
3. Train model: `train_attention_fast.py` with appropriate args
4. Evaluate: `evaluate_model.py` on test set
5. Analyze errors: Use `error_analysis.ipynb`
6. Iterate: Adjust architecture/hyperparameters

### **Running on Colab**
```bash
# Always use UV for consistency
!uv run python scripts/SCRIPT_NAME.py --args

# Save checkpoints to Drive
--checkpoint_dir /content/drive/MyDrive/vqa-checkpoints-NAME

# Use A100 GPU for speed
--batch_size 48 --device cuda
```

---

## **Outstanding Questions & Decisions**

### **1. RAG: Continue or Skip?**
- Current improved RAG is near baseline but unstable (66.54% to 68.46%)
- Need ablation clarity before deciding final direction

**Recommendation:** Run ablation matrix tomorrow; if best run stays <69%, switch focus to captioning.

### **2. What to Implement Next?**
**Options:**
- A) Finish RAG ablation and lock best settings
- B) Implement image captioning (likely higher impact)
- C) Adversarial prompting (test robustness)
- D) Leverage more GPU memory (bigger models/batches)

**Recommendation:** RAG ablation (short) → captioning (main path)

### **3. Target Accuracy?**
- Current: 69.07%
- Realistic target: 75-80% (competitive with papers)
- Stretch goal: 80%+

---

## **Code Patterns & Conventions**

### **Import Pattern**
```python
from vqa_med.models import AttentionVQAModel
from vqa_med.data import MedicalVQADataset
from vqa_med.utils import get_image_transforms, get_tokenizer
from vqa_med.config import config
```

### **Training Script Pattern**
```python
# 1. Parse args
args = parse_args()

# 2. Load data
dataset = MedicalVQADataset(data_csv, image_dir, transform, tokenizer)
train_loader, val_loader = create_loaders(dataset)

# 3. Create model
model = AttentionVQAModel(num_classes=dataset.get_num_classes())

# 4. Create trainer
trainer = Trainer(model, train_loader, val_loader, ...)

# 5. Train
trainer.train()
```

### **Colab Cell Pattern**
```python
# Cell 1: Setup
!git clone ... && cd vqa-med && uv pip install -e .

# Cell 2: Train
!uv run python scripts/train_X.py --args

# Cell 3: Evaluate
!uv run python scripts/evaluate_model.py --checkpoint ...
```

---

## **Contact & Continuation**

**Developer:** Radu  
**Project Type:** Research/Learning project  
**Goal:** Build progressively better medical VQA system  
**Current Best:** 69.07% with cross-attention model  
**Next Milestone:** 75%+ with multimodal enhancements  

**To continue:**
1. Read this summary completely
2. Check TODO.md for detailed next steps
3. Load best checkpoint and evaluate to confirm baseline
4. Choose next enhancement (captioning recommended)
5. Implement, train, evaluate, iterate

---

## **Quick Reference Commands**

```bash
# Dataset prep
!uv run python scripts/prepare_vqa_rad.py

# Analyze dataset
!uv run python scripts/analyze_dataset.py

# Train attention model (current best)
!uv run python scripts/train_attention_fast.py \
  --batch_size 48 --num_epochs 40 \
  --checkpoint_dir /content/drive/MyDrive/vqa-checkpoints-attention

# Evaluate model
!uv run python scripts/evaluate_model.py \
  --checkpoint /path/to/checkpoint_best.pth \
  --split test

# Build knowledge base for RAG
!uv run python scripts/build_knowledge_base.py --use_sample

# Train improved RAG model (stable config for tomorrow)
!uv run python scripts/train_rag_improved.py \
  --data_path data/processed/vqa_rad_closed.csv \
  --image_dir data/raw/VQA-RAD/images \
  --batch_size 12 \
  --gradient_accumulation_steps 2 \
  --learning_rate 5e-4 \
  --num_epochs 40 \
  --checkpoint_dir /content/drive/MyDrive/vqa-checkpoints-rag-stable \
  --device cuda \
  --visual_weight 0.25 \
  --temperature_init 1.0 \
  --top_k_docs 3 \
  --use_gated_fusion

# Train caption-augmented model (active path)
!uv run python scripts/train_caption.py \
  --data_csv data/processed/vqa_rad_closed.csv \
  --image_dir data/raw/VQA-RAD/images \
  --generate_captions \
  --caption_cache_file /content/drive/MyDrive/vqa-med-data/vqa_rad_closed_with_captions.csv \
  --caption_column caption \
  --caption_batch_size 8 \
  --batch_size 12 \
  --gradient_accumulation_steps 2 \
  --learning_rate 5e-4 \
  --num_epochs 40 \
  --checkpoint_dir /content/drive/MyDrive/vqa-checkpoints-caption \
  --device cuda

# Caption ablation run (tomorrow start here)
!uv run python scripts/train_caption.py \
  --data_csv /content/drive/MyDrive/vqa-med-data/vqa_rad_closed_with_captions.csv \
  --image_dir data/raw/VQA-RAD/images \
  --caption_column caption \
  --caption_max_length 48 \
  --batch_size 12 \
  --gradient_accumulation_steps 2 \
  --learning_rate 2e-4 \
  --num_epochs 40 \
  --checkpoint_dir /content/drive/MyDrive/vqa-checkpoints-caption-lr2e4 \
  --device cuda
```

---

## **Tomorrow Session Handoff (Copy/Paste Checklist)**

1. Keep the attention baseline (69.07%) as the decision threshold.
2. Continue captioning from cached captions first (no regeneration during train).
3. Run LR + caption-length ablations one at a time and log best val accuracy for each run.
4. Stop caption tuning if best run remains <69.07 after controlled ablations.
5. If stopped, run one final stable RAG sanity run and choose the stronger path.
6. Update this file with: command used, best val acc, checkpoint path, and next decision.

---

**End of Summary**
