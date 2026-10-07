from clause_extractor import (
    build_clause_reference_stats,
    flag_unusual_clauses,
    load_reference_stats,
    missing_expected_clauses,
    save_reference_stats,
)


def test_outlier_is_flagged_with_reason():
    vectors = {
        "standard one": [1.0, 0.0],
        "standard two": [0.99, 0.1],
        "standard three": [0.98, 0.2],
        "unusual": [0.0, 1.0],
    }
    clauses = [
        {"clause_text": text, "clause_type": "confidentiality"}
        for text in vectors
    ]
    result = flag_unusual_clauses(clauses, embedder=vectors.__getitem__)
    assert result[-1]["flagged_unusual"] is True
    assert result[-1]["flag_reason"] == "atypical wording for a confidentiality clause"


def test_similar_clause_is_not_flagged():
    vectors = {"one": [1.0, 0.0], "two": [1.0, 0.0], "three": [1.0, 0.0]}
    result = flag_unusual_clauses(
        [{"clause_text": text, "clause_type": "confidentiality"} for text in vectors],
        embedder=vectors.__getitem__,
    )
    assert all(clause["flagged_unusual"] is False for clause in result)
    assert all(clause["flag_reason"] is None for clause in result)


def test_missing_expected_clause():
    clauses = [{"clause_type": "termination"}]
    assert missing_expected_clauses(
        "nda", clauses, expected_clauses={"nda": {"termination", "confidentiality"}}
    ) == ["confidentiality"]


def test_no_expected_clause_is_missing():
    clauses = [
        {"clause_type": "termination"},
        {"clause_type": "confidentiality"},
    ]
    assert missing_expected_clauses(
        "nda", clauses, expected_clauses={"nda": {"termination", "confidentiality"}}
    ) == []


def test_reference_stats_flagging_and_round_trip(tmp_path):
    vectors = {
        f"normal-{index}": [1.0, 0.0] for index in range(5)
    }
    vectors.update({"unusual": [0.0, 1.0], "candidate": [0.0, 1.0]})
    training = {
        "confidentiality": [
            {"clause_text": name} for name in list(vectors)[:5]
        ]
    }
    stats = build_clause_reference_stats(training, vectors.__getitem__)
    path = tmp_path / "reference_stats.json"
    save_reference_stats(path, stats)
    loaded = load_reference_stats(path)
    result = flag_unusual_clauses(
        [{"clause_text": "candidate", "clause_type": "confidentiality"}],
        embedder=vectors.__getitem__,
        reference_stats=loaded,
    )
    assert result[0]["flagged_unusual"] is True


def test_reference_stats_with_fewer_than_five_examples_is_not_flagged():
    vectors = {f"clause-{index}": [1.0, 0.0] for index in range(4)}
    stats = build_clause_reference_stats(
        {"confidentiality": [{"clause_text": key} for key in vectors]},
        vectors.__getitem__,
    )
    result = flag_unusual_clauses(
        [{"clause_text": "clause-0", "clause_type": "confidentiality"}],
        embedder=vectors.__getitem__,
        reference_stats=stats,
    )
    assert result[0]["flagged_unusual"] is False
