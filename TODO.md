# VQA-Med Continuation Checklist

## Current milestone
- Reproduce the improved RAG best run at 70.38% validation accuracy.
- Validate the improved KB source quality first: topic relevance and duplicate rate.
- Treat caption at 69.59% as the current comparison baseline until RAG is reproduced.

## Latest robustness check (April 20, 2026)
- Caption clean accuracy: 63.78%
- Caption adversarial accuracy: 63.27%
- Robustness drop: 0.51 percentage points
- Interpretation: robustness is acceptable (small drop), but clean performance is below the previous caption high-water mark.

## Required run order
1. `scripts/prepare_vqa_rad.py`
2. `scripts/build_pmc_source_file.py`
3. `scripts/build_knowledge_base.py`
4. `scripts/train_rag_improved.py`
5. `scripts/train_caption.py`
6. `scripts/evaluate_model.py`

## Pending items
- Recreate the PMC source JSON with strict topic filtering.
- Build the improved KB and inspect the quality report.
- Reproduce the 70.38% improved-RAG run with fixed seeds.
- Run controlled comparison arms for standard RAG and captioning.
- Add adversarial prompting / robustness experiments after the clean run is stable. (initial check completed; continue after clean performance is re-raised)
- Compile a compact impact table with command, config diff, validation accuracy, and robustness notes.
- Choose the next experiment using mean accuracy and run-to-run variance.

## Locked reference points
- Attention baseline: 69.07%
- Caption historical best: 69.59%
- Improved RAG best: 70.38%
