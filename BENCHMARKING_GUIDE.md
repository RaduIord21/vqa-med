# Benchmarking Guide

This guide explains how to benchmark the current VQA models and collect the largest useful set of metrics and artifacts for thesis work.

## What The Benchmark Tool Does

The benchmark entry point is [scripts/benchmark_models.py](scripts/benchmark_models.py).

It can compare these model families on the same split:

* Baseline VQA model
* RAG-enhanced VQA model
* Caption-augmented VQA model
* KG-enhanced slot, currently skipped because no KG model is implemented yet

For each model, the benchmark reports:

* Accuracy
* Macro F1
* Weighted F1
* Per-class F1
* Sample count
* Status (`ok`, `skipped`, or `error`)

When you provide `--output_dir`, it also saves inference artifacts per model:

* `benchmark_results.json`
* `benchmark_results.csv`
* `inference.csv` with question, prediction, label, and correctness columns
* `confusion_matrix.json` with labels and matrix values
* `per_class_f1.json` with class-wise F1 values

## Recommended Thesis Workflow

Use the benchmark tool first to establish a fair comparison on the same split and seed. Then save the detailed artifacts for analysis, plotting, and figure generation.

Recommended order:

1. Benchmark baseline, RAG, and caption models on the validation split.
2. Save the outputs to a dedicated run directory.
3. Use the saved CSV and JSON files to generate plots for the thesis.
4. Repeat on the test split only after the model selection is finalized.

## Example Command

```bash
uv run python scripts/benchmark_models.py \
  --data_csv /content/drive/MyDrive/vqa-med-data/vqa_rad_closed_with_captions.csv \
  --image_dir data/raw/VQA-RAD/images \
  --baseline_checkpoint /content/drive/MyDrive/vqa-checkpoints-attention-fast/checkpoint_best.pth \
  --rag_checkpoint /content/drive/MyDrive/vqa-checkpoints-rag-stable/checkpoint_best.pth \
  --caption_checkpoint /content/drive/MyDrive/vqa-checkpoints-caption-lr1e4-len48/checkpoint_best.pth \
  --knowledge_base_path data/knowledge/medical_kb \
  --caption_column caption \
  --caption_max_length 48 \
  --split val \
  --batch_size 12 \
  --device cuda \
  --output_dir outputs/benchmarks/run_001
```

## What Each Argument Means

* `--data_csv`: processed dataset CSV. Use the captioned CSV if you want to benchmark the caption model.
* `--image_dir`: path to the image folder.
* `--baseline_checkpoint`: checkpoint for the baseline model.
* `--rag_checkpoint`: checkpoint for the RAG model.
* `--caption_checkpoint`: checkpoint for the caption model.
* `--knowledge_base_path`: required for RAG benchmarking.
* `--caption_column`: caption column name in the CSV.
* `--caption_max_length`: caption token length used by the caption model.
* `--split`: `train`, `val`, or `test`. Use `val` for iteration, `test` only for the final report.
* `--output_dir`: where benchmark results and artifacts are written.

## Artifact Layout

If `--output_dir outputs/benchmarks/run_001` is used, the files are written like this:

```text
outputs/benchmarks/run_001/
├── benchmark_results.json
├── benchmark_results.csv
├── baseline/
│   ├── inference.csv
│   ├── confusion_matrix.json
│   └── per_class_f1.json
├── rag/
│   ├── inference.csv
│   ├── confusion_matrix.json
│   └── per_class_f1.json
├── caption/
│   ├── inference.csv
│   ├── confusion_matrix.json
│   └── per_class_f1.json
└── kg-enhanced/
    └── skipped unless a KG model is added later
```

## Thesis-Ready Visuals You Can Derive

The benchmark already gives you the raw ingredients for visualizations. For thesis figures, build these from the saved artifacts:

* Model comparison bar chart for accuracy
* Model comparison bar chart for macro F1
* Model comparison bar chart for weighted F1
* Per-class F1 heatmap for each model
* Confusion matrix heatmap for each model
* Correct vs incorrect prediction table for qualitative examples
* Error case table sorted by confidence or by frequent confusion pairs

## Best Practice For Clean Comparisons

* Keep the dataset split seed fixed.
* Keep the same split for all models in one benchmark run.
* Use the same `batch_size` and `device` where possible.
* Benchmark the exact checkpoint you intend to report.
* Run the validation split repeatedly while tuning, and reserve the test split for the final locked result.

## Notes On The KG Slot

The benchmark has a KG placeholder, but it is currently skipped because the repository does not contain a KG-enhanced model implementation yet. Once that model exists, it can be added to the same benchmark workflow and will automatically get the same metrics and artifacts.

## Suggested Thesis Workflow

For the most complete set of artifacts:

1. Run the benchmark on the validation split with `--output_dir` set.
2. Copy the resulting CSV and JSON files into a thesis analysis folder.
3. Generate plots from the benchmark outputs.
4. Use the saved `inference.csv` files to pick representative examples for qualitative discussion.
5. Repeat the final locked benchmark on the test split once model selection is done.

## Short Version

If you only want the command, use:

```bash
uv run python scripts/benchmark_models.py --output_dir outputs/benchmarks/run_001
```

Then add the checkpoint flags for the models you want to evaluate.