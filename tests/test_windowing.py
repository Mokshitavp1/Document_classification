from evaluate import span_metrics
from windowing import assign_bio_labels, merge_predicted_spans


def test_continuation_window_gets_i_label():
    labels = {"O": 0, "B-confidentiality": 1, "I-confidentiality": 2}
    assert assign_bio_labels(
        [(10, 15), (15, 20)],
        [{"clause_type": "confidentiality", "span_start": 5, "span_end": 20}],
        labels,
    ) == [2, 2]


def test_merge_overlapping_window_predictions():
    merged = merge_predicted_spans(
        [
            {
                "document_name": "doc",
                "clause_type": "confidentiality",
                "clause_text": "abc",
                "source_text": "abcdefgh",
                "span_start": 0,
                "span_end": 3,
                "confidence_score": 0.9,
            },
            {
                "document_name": "doc",
                "clause_type": "confidentiality",
                "clause_text": "cdef",
                "source_text": "abcdefgh",
                "span_start": 2,
                "span_end": 6,
                "confidence_score": 0.7,
            },
        ]
    )
    assert len(merged) == 1
    assert merged[0]["span_start"] == 0
    assert merged[0]["span_end"] == 6
    assert merged[0]["confidence_score"] == 0.7


def test_span_metrics_exact_and_lenient():
    gold = [{"clause_type": "termination", "span_start": 10, "span_end": 20}]
    predicted = [{"clause_type": "termination", "span_start": 12, "span_end": 22}]
    metrics = span_metrics(gold, predicted)
    assert metrics["exact"]["macro"]["f1"] == 0.0
    assert metrics["lenient_iou_0.5"]["macro"]["f1"] == 1.0
