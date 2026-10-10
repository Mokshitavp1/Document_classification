from dataset_prep import determine_document_type


def test_document_type_matching_uses_word_boundaries():
    assert determine_document_type("agenda for review") is None
    assert determine_document_type("fundamental terms") is None
    assert determine_document_type("Acme Document") is None
    assert determine_document_type("Acme NDA") == "nda"
    assert determine_document_type("Acme IP License") == "ip"


def test_document_type_prefers_specific_signals_over_generic_agreement():
    assert determine_document_type("Acme Employment Agreement") == "employment"
    assert determine_document_type("Acme Lease Agreement") == "rental"
    assert determine_document_type("Acme Confidentiality Agreement") == "nda"


def test_document_type_does_not_use_substring_matches():
    assert determine_document_type("Acme NDAA Agreement") == "service"
    assert determine_document_type("Acme Serviceable Contract") == "service"
