"""Inference API for the fine-tuned Legal-BERT clause model."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification, AutoTokenizer

from contract_constants import CLAUSE_TYPES, DOCUMENT_TYPES
from windowing import merge_predicted_spans

CHECKPOINT_PATH = Path(os.environ.get("CLASSIFIER_CHECKPOINT_PATH", "checkpoints/legal_bert_clause_v1"))
_tokenizer = None
_document_model = None
_clause_model = None


def _load_models() -> tuple[Any, Any, Any]:
    global _tokenizer, _document_model, _clause_model
    if _tokenizer is None:
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at {CHECKPOINT_PATH}. Run train.py first."
            )
        _tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
        _document_model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_PATH / "document_classifier")
        _clause_model = AutoModelForTokenClassification.from_pretrained(CHECKPOINT_PATH / "clause_extractor")
        _document_model.eval()
        _clause_model.eval()
    return _tokenizer, _document_model, _clause_model


def classify_document(text: str, document_name: str = "uploaded_document") -> dict[str, Any]:
    tokenizer, model, _ = _load_models()
    # Document classification intentionally uses only the first 512 tokens.
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.inference_mode():
        probabilities = torch.softmax(model(**inputs).logits, dim=-1)[0]
    index = int(probabilities.argmax())
    label = model.config.id2label[index]
    if label not in DOCUMENT_TYPES:
        raise ValueError(f"Checkpoint returned invalid document_type: {label!r}")
    return {
        "document_name": document_name,
        "document_type": label,
        "confidence_score": float(probabilities[index]),
    }


def extract_clauses(text: str, document_name: str = "uploaded_document") -> list[dict[str, Any]]:
    tokenizer, _, model = _load_models()
    encoded = tokenizer(
        text, return_offsets_mapping=True, return_special_tokens_mask=True,
        return_overflowing_tokens=True, stride=128, padding=True,
        return_tensors="pt", truncation=True, max_length=512,
    )
    offsets = encoded.pop("offset_mapping").tolist()
    encoded.pop("special_tokens_mask", None)
    encoded.pop("overflow_to_sample_mapping", None)
    window_count = encoded["input_ids"].shape[0]
    with torch.inference_mode():
        predictions = model(**encoded).logits
    probabilities = torch.softmax(predictions, dim=-1)
    window_spans: list[dict[str, Any]] = []
    for window_index in range(window_count):
        active: dict[str, Any] | None = None
        labels = probabilities[window_index].argmax(dim=-1).tolist()
        for offset, label_index, scores in zip(offsets[window_index], labels, probabilities[window_index]):
            start, end = offset
            label = model.config.id2label[label_index]
            if label == "O" or start == end:
                if active:
                    window_spans.append(active)
                    active = None
                continue
            clause_type = label.removeprefix("B-").removeprefix("I-")
            if clause_type not in CLAUSE_TYPES:
                continue
            if active is None or label.startswith("B-") or active["clause_type"] != clause_type:
                if active:
                    window_spans.append(active)
                active = {
                    "clause_text": text[start:end],
                    "source_text": text,
                    "clause_type": clause_type,
                    "document_name": document_name,
                    "span_start": start,
                    "span_end": end,
                    "confidence_score": float(scores[label_index]),
                }
            else:
                active["span_end"] = end
                active["clause_text"] = text[active["span_start"]:end]
                active["confidence_score"] = min(
                    active["confidence_score"], float(scores[label_index])
                )
        if active:
            window_spans.append(active)
    merged = merge_predicted_spans(window_spans)
    for span in merged:
        span.pop("source_text", None)
        span["clause_text"] = text[span["span_start"]:span["span_end"]]
    return merged
