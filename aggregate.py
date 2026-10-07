"""Assembly of the contract-defined document-level result."""

from collections.abc import Iterable, Mapping
from typing import Any

from clause_extractor import EXPECTED_CLAUSES, missing_expected_clauses


def aggregate_document(
    document_name: str,
    classification: Mapping[str, Any],
    clauses: Iterable[Mapping[str, Any]],
    *,
    expected_clauses: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, Any]:
    """Build a 4.5 result from a 4.1 classification and 4.4 clauses."""
    if classification.get("document_name", document_name) != document_name:
        raise ValueError("classification document_name does not match document_name")
    if "document_type" not in classification or "confidence_score" not in classification:
        raise ValueError("classification must contain document_type and confidence_score")

    clause_list = [dict(clause) for clause in clauses]
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
            expected_clauses=expected_clauses or EXPECTED_CLAUSES,
        ),
    }
