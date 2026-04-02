"""Benchmark baseline, KG, RAG, and caption models on the same split."""

from __future__ import annotations

import argparse
from pathlib import Path

from vqa_med.benchmarking import BenchmarkRunner, build_default_specs
from vqa_med.config import config


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Medical VQA models")
    parser.add_argument("--data_csv", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--caption_column", type=str, default="caption")
    parser.add_argument("--caption_max_length", type=int, default=48)
    parser.add_argument("--knowledge_base_path", type=str, default=None)
    parser.add_argument("--baseline_checkpoint", type=str, default=None)
    parser.add_argument("--kg_checkpoint", type=str, default=None)
    parser.add_argument("--rag_checkpoint", type=str, default=None)
    parser.add_argument("--caption_checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--top_k_docs", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    data_csv = Path(args.data_csv) if args.data_csv else config.paths.processed_data / "vqa_rad_closed_with_captions.csv"
    image_dir = Path(args.image_dir) if args.image_dir else config.paths.raw_data / "VQA-RAD" / "images"
    knowledge_base_path = Path(args.knowledge_base_path) if args.knowledge_base_path else None

    specs = build_default_specs(
        data_csv=data_csv,
        image_dir=image_dir,
        baseline_checkpoint=Path(args.baseline_checkpoint) if args.baseline_checkpoint else None,
        kg_checkpoint=Path(args.kg_checkpoint) if args.kg_checkpoint else None,
        rag_checkpoint=Path(args.rag_checkpoint) if args.rag_checkpoint else None,
        caption_checkpoint=Path(args.caption_checkpoint) if args.caption_checkpoint else None,
        knowledge_base_path=knowledge_base_path,
        caption_column=args.caption_column,
        caption_max_length=args.caption_max_length,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        split=args.split,
        top_k_docs=args.top_k_docs,
    )

    output_dir = Path(args.output_dir) if args.output_dir else None
    runner = BenchmarkRunner(specs, output_dir=output_dir)
    results = runner.run()

    for result in results:
        accuracy = f"{result.accuracy:.2f}%" if result.accuracy is not None else "N/A"
        macro_f1 = f"{result.macro_f1:.4f}" if result.macro_f1 is not None else "N/A"
        weighted_f1 = f"{result.weighted_f1:.4f}" if result.weighted_f1 is not None else "N/A"
        print(
            f"{result.name}: {result.status} | acc={accuracy} | macro_f1={macro_f1} | "
            f"weighted_f1={weighted_f1} | samples={result.num_samples} | {result.notes}"
        )

    if args.output_dir:
        runner.save_report(results, output_dir)
        print(f"Saved benchmark report to {args.output_dir}")


if __name__ == "__main__":
    main()

