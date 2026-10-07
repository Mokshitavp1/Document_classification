"""Assembly of the contract-defined document-level result."""

from collections.abc import Iterable, Mapping
import json
import os
from pathlib import Path
from typing import Any

from clause_extractor import EXPECTED_CLAUSES, missing_expected_clauses


def aggregate_document(
    document_name: str,
    classification: Mapping[str, Any],
    clauses: Iterable[Mapping[str, Any]],
    *,
    expected_clauses: Mapping[str, Iterable[str]] | None = None,
    expected_clauses_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a 4.5 result from a 4.1 classification and 4.4 clauses."""
    if classification.get("document_name", document_name) != document_name:
        raise ValueError("classification document_name does not match document_name")
    if "document_type" not in classification or "confidence_score" not in classification:
        raise ValueError("classification must contain document_type and confidence_score")

    clause_list = [dict(clause) for clause in clauses]
    mapping = expected_clauses
    if mapping is None:
        candidate = Path(expected_clauses_path) if expected_clauses_path else None
        if candidate is None:
            checkpoint = os.environ.get("CLASSIFIER_CHECKPOINT_PATH")
            if checkpoint:
                candidate = Path(checkpoint) / "expected_clauses.json"
        if candidate and candidate.exists():
            mapping = json.loads(candidate.read_text(encoding="utf-8"))
            mapping = {key: set(value) for key, value in mapping.items()}
    return {
        "document_name": document_name,
        "classification": {
            "document_type": classification["document_type"],
            "confidence_score": classification["confidence_score"],
        },
        "clauses": clause_list,
        "missing_expected_clauses": missing_expected_clauses(
            classification["document_type"],
            clause_list,
            expected_clauses=mapping or EXPECTED_CLAUSES,
        ),
    }
