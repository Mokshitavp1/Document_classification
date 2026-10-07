"""Post-process model clauses with explainable outlier and omission checks."""

from collections.abc import Callable, Iterable, Mapping, Sequence
from math import sqrt
from typing import Any

from contract_constants import CLAUSE_TYPES, DOCUMENT_TYPES

EmbeddingFn = Callable[[str], Sequence[float]]

# This is intentionally replaceable with the corpus-derived mapping from the
# trained model module. It is kept small until that corpus is available.
EXPECTED_CLAUSES: dict[str, set[str]] = {
    "nda": {"confidentiality", "termination"},
    "employment": {"termination", "confidentiality"},
}


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding vectors must have the same dimension")
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _default_embedder() -> EmbeddingFn:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for production outlier detection; "
            "install it or pass an embedder explicitly"
        ) from exc

    model = SentenceTransformer("all-MiniLM-L6-v2")
    return lambda text: model.encode(text, normalize_embeddings=False).tolist()


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile from no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def flag_unusual_clauses(
    clauses: Iterable[Mapping[str, Any]],
    *,
    embedder: EmbeddingFn | None = None,
    percentile: float = 0.10,
) -> list[dict[str, Any]]:
    """Add the 4.4 unusual-clause fields to model clause dictionaries.

    Similarity is measured against the centroid of all *other* clauses with
    the same type. A type with fewer than two examples has no comparison set
    and is not flagged.
    """
    if not 0 <= percentile <= 1:
        raise ValueError("percentile must be between 0 and 1")

    result = [dict(clause) for clause in clauses]
    for clause in result:
        clause_type = clause.get("clause_type")
        if clause_type not in CLAUSE_TYPES:
            raise ValueError(f"Unknown clause_type: {clause_type!r}")

    embed = embedder or _default_embedder()
    vectors = [embed(clause["clause_text"]) for clause in result]
    by_type: dict[str, list[int]] = {}
    for index, clause in enumerate(result):
        by_type.setdefault(clause["clause_type"], []).append(index)

    for index, clause in enumerate(result):
        peers = [peer for peer in by_type[clause["clause_type"]] if peer != index]
        flagged = False
        if peers:
            dimensions = len(vectors[index])
            if dimensions == 0 or any(len(vectors[peer]) != dimensions for peer in peers):
                raise ValueError("All embeddings must be non-empty and have equal dimensions")
            centroid = [
                sum(vectors[peer][dimension] for peer in peers) / len(peers)
                for dimension in range(dimensions)
            ]
            similarity = _cosine_similarity(vectors[index], centroid)
            peer_similarities = [
                _cosine_similarity(vectors[peer], centroid) for peer in peers
            ]
            threshold = _percentile([similarity, *peer_similarities], percentile)
            flagged = similarity < threshold
        clause["flagged_unusual"] = flagged
        clause["flag_reason"] = (
            f"atypical wording for a {clause['clause_type']} clause" if flagged else None
        )
    return result


def missing_expected_clauses(
    document_type: str,
    clauses: Iterable[Mapping[str, Any]],
    *,
    expected_clauses: Mapping[str, Iterable[str]] | None = None,
) -> list[str]:
    """Return expected clause types absent from a document, in enum order."""
    if document_type not in DOCUMENT_TYPES:
        raise ValueError(f"Unknown document_type: {document_type!r}")
    mapping = expected_clauses or EXPECTED_CLAUSES
    expected = set(mapping.get(document_type, ()))
    found = {clause["clause_type"] for clause in clauses}
    invalid = expected - set(CLAUSE_TYPES)
    if invalid:
        raise ValueError(f"Unknown expected clause types: {sorted(invalid)}")
    return [clause_type for clause_type in CLAUSE_TYPES if clause_type in expected - found]


def build_expected_clauses(
    documents: Iterable[Mapping[str, Any]], *, minimum_presence: float = 0.5
) -> dict[str, set[str]]:
    """Derive expected clauses from document-level co-occurrence statistics."""
    if not 0 <= minimum_presence <= 1:
        raise ValueError("minimum_presence must be between 0 and 1")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for document in documents:
        grouped.setdefault(document["document_type"], []).append(document)
    result: dict[str, set[str]] = {}
    for document_type, group in grouped.items():
        counts = {clause_type: 0 for clause_type in CLAUSE_TYPES}
        for document in group:
            present = {clause["clause_type"] for clause in document.get("clauses", ())}
            for clause_type in present:
                if clause_type in counts:
                    counts[clause_type] += 1
        result[document_type] = {
            clause_type
            for clause_type, count in counts.items()
            if count / len(group) >= minimum_presence
        }
    return result
