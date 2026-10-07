"""Small, deterministic fixtures for integration tests before model delivery."""

SAMPLE_CLASSIFICATION = {
    "document_name": "sample_nda_001",
    "document_type": "nda",
    "confidence_score": 0.91,
}

SAMPLE_CLAUSES = [
    {
        "clause_text": "The Receiving Party shall keep all Confidential Information strictly confidential.",
        "clause_type": "confidentiality",
        "document_name": "sample_nda_001",
        "span_start": 34,
        "span_end": 112,
        "confidence_score": 0.88,
    },
    {
        "clause_text": "This Agreement shall terminate upon thirty days written notice.",
        "clause_type": "termination",
        "document_name": "sample_nda_001",
        "span_start": 125,
        "span_end": 186,
        "confidence_score": 0.86,
    },
]

SAMPLE_EXPECTED_CLAUSES = {
    "nda": {"confidentiality", "termination"},
    "employment": {"termination", "confidentiality"},
}
