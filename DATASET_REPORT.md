# Dataset report

## Document counts per type

| Document type | Count |
|---|---:|
| employment | 0 |
| rental | 0 |
| privacy | 0 |
| ip | 50 |
| service | 429 |
| nda | 1 |

## Clause counts per type

| Clause type | Count |
|---|---:|
| termination | 235 |
| confidentiality | 0 |
| indemnification | 0 |
| governing_law | 447 |
| payment_terms | 0 |
| liability_limitation | 809 |
| assignment | 633 |
| non_compete | 248 |
| non_solicitation | 134 |
| ip_ownership | 302 |
| data_processing | 0 |
| renewal | 204 |
| dispute_resolution | 0 |
| force_majeure | 0 |
| warranty | 160 |

## Missing and low-coverage clause types

- Zero examples: confidentiality, indemnification, payment_terms, data_processing, dispute_resolution, force_majeure
- Fewer than 20 examples: None

## Changes

- Replaced the obsolete script-based CUAD loader with the official JSON export loaded through the built-in JSON dataset builder, preserving all contracts.
- Flattened CUAD's nested paragraphs and QA answers into the shared document shape and skipped only answers whose supplied offsets do not match their context.
- Matched the real 41 CUAD question categories, including liability caps, uncapped liability, warranty duration, non-solicitation, renewal, assignment, IP ownership, governing law, and termination for convenience.
- Expanded document-type detection with word-boundary-only keyword patterns.
