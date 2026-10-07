from aggregate import aggregate_document
from doc_analysis_fixtures import SAMPLE_CLASSIFICATION, SAMPLE_CLAUSES


def test_aggregate_reports_missing_expected_clause():
    result = aggregate_document(
        "sample_nda_001",
        SAMPLE_CLASSIFICATION,
        [SAMPLE_CLAUSES[0]],
        expected_clauses={"nda": {"termination", "confidentiality"}},
    )
    assert result["document_name"] == "sample_nda_001"
    assert result["classification"] == {
        "document_type": "nda",
        "confidence_score": 0.91,
    }
    assert result["missing_expected_clauses"] == ["termination"]


def test_aggregate_reports_no_missing_clauses():
    result = aggregate_document(
        "sample_nda_001",
        SAMPLE_CLASSIFICATION,
        SAMPLE_CLAUSES,
        expected_clauses={"nda": {"termination", "confidentiality"}},
    )
    assert result["missing_expected_clauses"] == []
    assert result["clauses"] == SAMPLE_CLAUSES


def test_aggregate_loads_expected_clauses_from_path(tmp_path):
    path = tmp_path / "expected_clauses.json"
    path.write_text('{"nda": ["confidentiality"]}', encoding="utf-8")
    result = aggregate_document(
        "sample_nda_001",
        SAMPLE_CLASSIFICATION,
        [SAMPLE_CLAUSES[0]],
        expected_clauses_path=path,
    )
    assert result["missing_expected_clauses"] == []
