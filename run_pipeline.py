"""
End-to-end pipeline runner.

Runs in order:
    1. generate_splits  — create splits.json (70/15/15 stratified)
    2. train            — fine-tune document classifier + clause extractor
    3. build_artifacts  — compute reference_stats.json + expected_clauses.json
                          from the training split only
    4. evaluate         — run held-out test evaluation; write evaluation_results.json

Usage:
    python run_pipeline.py [--epochs N] [--max-train-docs N] [--baseline tfidf]

Each step is idempotent where possible:
  - splits.json is only written if it does not already exist.
  - Training always overwrites the checkpoint directory (use --output to pin a version).
  - Evaluation results are always overwritten.

Environment variables:
    CLASSIFIER_CHECKPOINT_PATH  Override default checkpoint path (default: checkpoints/legal_bert_clause_v1)
    OPENAI_API_KEY              Required only if running bootstrap_pipeline.py separately.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _run(args: list[str], description: str) -> None:
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable] + args, check=False)
    if result.returncode != 0:
        print(f"\n❌  Step failed (exit {result.returncode}): {description}")
        sys.exit(result.returncode)
    print(f"✅  Done: {description}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full document-classification pipeline end-to-end."
    )
    parser.add_argument(
        "--splits", type=Path, default=Path("splits.json"),
        help="Path to splits.json (created if absent).",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("checkpoints/legal_bert_clause_v1"),
        help="Checkpoint output directory.",
    )
    parser.add_argument("--epochs", type=float, default=3, help="Training epochs.")
    parser.add_argument(
        "--max-train-docs", type=int, default=None,
        help="Cap training documents (useful for smoke tests).",
    )
    parser.add_argument(
        "--baseline", choices=["tfidf"], default=None,
        help="Include a TF-IDF baseline in the evaluation report.",
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip training (use an existing checkpoint).",
    )
    args = parser.parse_args()

    # ── Step 1: generate splits ────────────────────────────────────────────────
    _run(["generate_splits.py"], "Generate train/val/test splits")

    # ── Step 2: train ─────────────────────────────────────────────────────────
    if not args.skip_train:
        train_cmd = [
            "train.py",
            "--splits", str(args.splits),
            "--output", str(args.output),
            "--epochs", str(args.epochs),
        ]
        if args.max_train_docs is not None:
            train_cmd += ["--max-train-docs", str(args.max_train_docs)]
        _run(train_cmd, "Fine-tune document classifier + clause extractor")
    else:
        print("\nSkipping training (--skip-train).")

    # ── Step 3: build reference artifacts ─────────────────────────────────────
    _run(
        [
            "clause_extractor.py",
            "bootstrap_labels_corrected.json",
            str(args.splits),
            str(args.output),
        ],
        "Build reference_stats.json + expected_clauses.json from training split",
    )

    # ── Step 4: evaluate ──────────────────────────────────────────────────────
    eval_cmd = ["evaluate.py", "--splits", str(args.splits)]
    if args.baseline:
        eval_cmd += ["--baseline", args.baseline]
    _run(eval_cmd, "Evaluate on held-out test split")

    # ── Summary ───────────────────────────────────────────────────────────────
    results_path = Path("evaluation_results.json")
    if results_path.exists():
        results = json.loads(results_path.read_text(encoding="utf-8"))
        cls_f1 = results.get("classification_macro_f1", "n/a")
        clause_f1 = (
            results.get("clause_spans_overall", {})
            .get("lenient_iou_0.5", {})
            .get("macro", {})
            .get("f1", "n/a")
        )
        print(f"\n{'='*60}")
        print("PIPELINE COMPLETE")
        print(f"  Classification macro-F1 : {cls_f1:.4f}" if isinstance(cls_f1, float) else f"  Classification macro-F1 : {cls_f1}")
        print(f"  Clause span F1 (IoU≥0.5): {clause_f1:.4f}" if isinstance(clause_f1, float) else f"  Clause span F1 (IoU≥0.5): {clause_f1}")
        print(f"  Full results: {results_path}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
