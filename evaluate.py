"""Held-out evaluation for document and clause predictions."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import make_pipeline

from dataset_prep import get_full_dataset
from legal_bert_model import classify_document, extract_clauses


def _span_key(span: Mapping[str, Any]) -> tuple[str, int, int]:
    return span["clause_type"], span["span_start"], span["span_end"]


def _iou(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    intersection = max(0, min(left["span_end"], right["span_end"]) - max(left["span_start"], right["span_start"]))
    union = max(left["span_end"], right["span_end"]) - min(left["span_start"], right["span_start"])
    return intersection / union if union else 0.0


def _prf(tp: int, predicted: int, gold: int) -> dict[str, float]:
    precision = tp / predicted if predicted else 0.0
    recall = tp / gold if gold else 0.0
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def span_metrics(
    gold_spans: Iterable[Mapping[str, Any]],
    predicted_spans: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute exact and IoU>=0.5 clause metrics by clause type."""
    gold = list(gold_spans)
    predicted = list(predicted_spans)
    types = sorted({span["clause_type"] for span in gold + predicted})
    exact_by_type: dict[str, dict[str, float]] = {}
    lenient_by_type: dict[str, dict[str, float]] = {}
    for clause_type in types:
        gold_type = [span for span in gold if span["clause_type"] == clause_type]
        predicted_type = [span for span in predicted if span["clause_type"] == clause_type]
        gold_keys = {_span_key(span) for span in gold_type}
        exact_tp = sum(_span_key(span) in gold_keys for span in predicted_type)
        lenient_tp = sum(
            any(_iou(predicted_span, gold_span) >= 0.5 for gold_span in gold_type)
            for predicted_span in predicted_type
        )
        exact_by_type[clause_type] = _prf(exact_tp, len(predicted_type), len(gold_type))
        lenient_by_type[clause_type] = _prf(lenient_tp, len(predicted_type), len(gold_type))
    return {
        "exact": {
            "by_clause_type": exact_by_type,
            "macro": _macro_metrics(exact_by_type),
        },
        "lenient_iou_0.5": {
            "by_clause_type": lenient_by_type,
            "macro": _macro_metrics(lenient_by_type),
        },
    }


def _macro_metrics(metrics: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    if not metrics:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        key: sum(item[key] for item in metrics.values()) / len(metrics)
        for key in ("precision", "recall", "f1")
    }


def evaluate_baseline(
    train_documents: list[Mapping[str, Any]],
    test_documents: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate a TF-IDF plus logistic-regression document classifier."""
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=1),
        LogisticRegression(max_iter=1000, random_state=42),
    )
    model.fit([doc["text"] for doc in train_documents], [doc["document_type"] for doc in train_documents])
    actual = [doc["document_type"] for doc in test_documents]
    predicted = model.predict([doc["text"] for doc in test_documents])
    return {
        "macro_f1": float(f1_score(actual, predicted, average="macro")),
        "classification_report": classification_report(actual, predicted, output_dict=True, zero_division=0),
    }


def evaluate(splits_path: str = "splits.json", baseline: str | None = None) -> dict[str, Any]:
    splits = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    all_documents = get_full_dataset()
    documents = {doc["document_name"]: doc for doc in all_documents}
    missing = [name for names in splits.values() for name in names if name not in documents]
    if missing:
        raise ValueError(f"Test or split documents missing from dataset, e.g. {missing[0]}")
    test_documents = [documents[name] for name in splits["test"]]
    actual_types: list[str] = []
    predicted_types: list[str] = []
    spans_by_type: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: {"gold": [], "predicted": []})
    for document in test_documents:
        classification = classify_document(document["text"], document["document_name"])
        predicted = extract_clauses(document["text"], document["document_name"])
        actual_types.append(document["document_type"])
        predicted_types.append(classification["document_type"])
        spans_by_type[document["document_type"]]["gold"].extend(document["clauses"])
        spans_by_type[document["document_type"]]["predicted"].extend(predicted)
    result: dict[str, Any] = {
        "classification_macro_f1": float(f1_score(actual_types, predicted_types, average="macro")),
        "classification_report": classification_report(actual_types, predicted_types, output_dict=True, zero_division=0),
        "clause_spans_overall": span_metrics(
            [span for group in spans_by_type.values() for span in group["gold"]],
            [span for group in spans_by_type.values() for span in group["predicted"]],
        ),
        "clause_spans_by_document_type": {
            document_type: span_metrics(group["gold"], group["predicted"])
            for document_type, group in spans_by_type.items()
        },
    }
    if baseline == "tfidf":
        result["baseline_tfidf"] = evaluate_baseline(
            [documents[name] for name in splits["train"]], test_documents
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", default="splits.json")
    parser.add_argument("--baseline", choices=["tfidf"])
    args = parser.parse_args()
    results = evaluate(args.splits, args.baseline)
    print(json.dumps(results, indent=2))
    Path("evaluation_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
