"""Post-process model clauses with explainable outlier and omission checks."""

from collections.abc import Callable, Iterable, Mapping, Sequence
import json
from math import sqrt
from pathlib import Path
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


def build_clause_reference_stats(
    training_clauses_by_type: Mapping[str, Iterable[Mapping[str, Any]]],
    embedder: EmbeddingFn,
    percentile: float = 0.10,
) -> dict[str, dict[str, Any]]:
    """Build train-corpus centroids and leave-one-out similarity thresholds."""
    if not 0 <= percentile <= 1:
        raise ValueError("percentile must be between 0 and 1")
    stats: dict[str, dict[str, Any]] = {}
    for clause_type, clauses in training_clauses_by_type.items():
        vectors = [list(embedder(clause["clause_text"])) for clause in clauses]
        if not vectors:
            continue
        dimensions = len(vectors[0])
        if dimensions == 0 or any(len(vector) != dimensions for vector in vectors):
            raise ValueError("All embeddings must be non-empty and have equal dimensions")
        centroid = [
            sum(vector[dimension] for vector in vectors) / len(vectors)
            for dimension in range(dimensions)
        ]
        similarities = []
        for index, vector in enumerate(vectors):
            if len(vectors) > 1:
                peers = [other for peer_index, other in enumerate(vectors) if peer_index != index]
                reference = [
                    sum(peer[dimension] for peer in peers) / len(peers)
                    for dimension in range(dimensions)
                ]
            else:
                reference = centroid
            similarities.append(_cosine_similarity(vector, reference))
        stats[clause_type] = {
            "centroid": centroid,
            "threshold": _percentile(similarities, percentile),
            "count": len(vectors),
        }
    return stats


def save_reference_stats(path: str | Path, stats: Mapping[str, Mapping[str, Any]]) -> None:
    Path(path).write_text(json.dumps(stats, indent=2), encoding="utf-8")


def load_reference_stats(path: str | Path) -> dict[str, dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Reference stats must be a JSON object")
    return data


def flag_unusual_clauses(
    clauses: Iterable[Mapping[str, Any]],
    *,
    embedder: EmbeddingFn | None = None,
    percentile: float = 0.10,
    reference_stats: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Add the 4.4 unusual-clause fields to model clause dictionaries.

    Reference stats use the training corpus. Without them, the within-document
    comparison is retained as a backward-compatible fallback.
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
        flagged = False
        if reference_stats is not None:
            reference = reference_stats.get(clause["clause_type"])
            if reference and int(reference.get("count", 0)) >= 5:
                similarity = _cosine_similarity(vectors[index], reference["centroid"])
                flagged = similarity < float(reference["threshold"])
        else:
            peers = [peer for peer in by_type[clause["clause_type"]] if peer != index]
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


def build_reference_artifacts(
    documents: Iterable[Mapping[str, Any]],
    splits: Mapping[str, Iterable[str]],
    embedder: EmbeddingFn,
    out_dir: str | Path,
) -> None:
    """Write reference stats and expected clauses using only train documents."""
    by_name = {document["document_name"]: document for document in documents}
    train_documents = [by_name[name] for name in splits["train"]]
    clauses_by_type: dict[str, list[Mapping[str, Any]]] = {}
    for document in train_documents:
        for clause in document.get("clauses", []):
            clauses_by_type.setdefault(clause["clause_type"], []).append(clause)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_reference_stats(output / "reference_stats.json", build_clause_reference_stats(clauses_by_type, embedder))
    expected = {
        document_type: sorted(clause_types)
        for document_type, clause_types in build_expected_clauses(train_documents).items()
    }
    (output / "expected_clauses.json").write_text(json.dumps(expected, indent=2), encoding="utf-8")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("documents_json", type=Path)
    parser.add_argument("splits_json", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    documents = json.loads(args.documents_json.read_text(encoding="utf-8"))
    splits = json.loads(args.splits_json.read_text(encoding="utf-8"))
    build_reference_artifacts(documents, splits, _default_embedder(), args.out_dir)
