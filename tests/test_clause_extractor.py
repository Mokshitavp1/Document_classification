from clause_extractor import flag_unusual_clauses, missing_expected_clauses


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
