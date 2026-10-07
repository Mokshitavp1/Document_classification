"""Pure helpers for sliding-window token labels and clause-span merging."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def assign_bio_labels(
    offsets: Sequence[Sequence[int]],
    clauses: Iterable[Mapping[str, Any]],
    label2id: Mapping[str, int],
    *,
    special_tokens_mask: Sequence[int] | None = None,
    attention_mask: Sequence[int] | None = None,
) -> list[int]:
    """Assign BIO ids to one tokenizer window using document character spans."""
    clause_list = list(clauses)
    labels: list[int] = []
    for index, offset in enumerate(offsets):
        start, end = int(offset[0]), int(offset[1])
        is_special = (
            start == end
            or special_tokens_mask is not None and special_tokens_mask[index] == 1
            or attention_mask is not None and attention_mask[index] == 0
        )
        if is_special:
            labels.append(-100)
            continue
        label = "O"
        for clause in clause_list:
            if start < clause["span_end"] and end > clause["span_start"]:
                prefix = "B" if start <= clause["span_start"] else "I"
                label = f"{prefix}-{clause['clause_type']}"
                break
        labels.append(label2id[label])
    return labels


def merge_predicted_spans(spans: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Merge overlapping or adjacent same-type window predictions."""
    ordered = sorted(
        (dict(span) for span in spans),
        key=lambda span: (span["document_name"], span["clause_type"], span["span_start"]),
    )
    merged: list[dict[str, Any]] = []
    for span in ordered:
        if (
            merged
            and merged[-1]["document_name"] == span["document_name"]
            and merged[-1]["clause_type"] == span["clause_type"]
            and span["span_start"] <= merged[-1]["span_end"]
        ):
            previous = merged[-1]
            previous["span_end"] = max(previous["span_end"], span["span_end"])
            previous["clause_text"] = span.get(
                "source_text", previous["clause_text"]
            )
            if "source_text" in span:
                previous["clause_text"] = span["source_text"][
                    previous["span_start"] : previous["span_end"]
                ]
            previous["confidence_score"] = min(
                previous["confidence_score"], span["confidence_score"]
            )
        else:
            merged.append(span)
    return merged
