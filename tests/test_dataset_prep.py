from dataset_prep import determine_document_type


def test_document_type_matching_uses_word_boundaries():
    assert determine_document_type("agenda for review") is None
    assert determine_document_type("fundamental terms") is None
    assert determine_document_type("Acme NDA") == "nda"
    assert determine_document_type("Acme IP License") == "ip"
