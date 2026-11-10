from dataset_prep import determine_document_type, is_excluded_from_training


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
    assert determine_document_type("Acme NDAA") is None
    assert determine_document_type("Acme Serviceable") is None


def test_document_type_unmatched_titles_return_none():
    assert determine_document_type("Meeting Minutes") is None
    assert determine_document_type("Press Release") is None
    assert determine_document_type("Acme Corporation") is None


def test_document_type_generic_keywords_match_as_service():
    assert determine_document_type("Acme NDAA Agreement") == "service"
    assert determine_document_type("Acme Serviceable Contract") == "service"
    assert determine_document_type("Standard Agreement") == "service"
    assert determine_document_type("Commercial Contract") == "service"


def test_dummy_documents_are_excluded_from_training():
    assert is_excluded_from_training("dummy_employment_001")
    assert not is_excluded_from_training("Acme Service Agreement")
