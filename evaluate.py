"""Held-out evaluation for document and clause predictions."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from seqeval.metrics import f1_score
from sklearn.metrics import f1_score as macro_f1

from dataset_prep import get_full_dataset
from legal_bert_model import classify_document, extract_clauses


def evaluate(splits_path: str = "splits.json") -> dict[str, Any]:
    splits = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    documents = {doc["document_name"]: doc for doc in get_full_dataset()}
    classification_true: list[str] = []
    classification_pred: list[str] = []
    clause_scores: dict[str, list[float]] = defaultdict(list)
    for name in splits["test"]:
        document = documents[name]
        classification = classify_document(document["text"], name)
        classification_true.append(document["document_type"])
        classification_pred.append(classification["document_type"])
        predicted = extract_clauses(document["text"], name)
        gold_labels = ["O"] * len(document["text"])
        pred_labels = ["O"] * len(document["text"])
        for clause in document["clauses"]:
            gold_labels[clause["span_start"]:clause["span_end"]] = [clause["clause_type"]] * (clause["span_end"] - clause["span_start"])
        for clause in predicted:
            pred_labels[clause["span_start"]:clause["span_end"]] = [clause["clause_type"]] * (clause["span_end"] - clause["span_start"])
        clause_scores[document["document_type"]].append(
            f1_score([gold_labels], [pred_labels], average="macro")
        )
    return {
        "classification_macro_f1": macro_f1(classification_true, classification_pred, average="macro"),
        "clause_span_f1_by_category": {key: sum(value) / len(value) for key, value in clause_scores.items()},
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
