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

## Title-based document-type facts

The title heuristic found 1 NDA, 0 employment, and 0 rental contracts. 166 contracts were labeled only by generic agreement/contract keywords. 30 contracts were excluded because their titles matched nothing.

## Changes and remaining problems

- Loaded only CUAD contracts whose titles matched a document-type keyword and skipped contracts whose titles matched nothing.
- Used word-boundary regexes with specific title signals before generic agreement and contract signals.
- Kept only CUAD categories that directly represent our clause types; confidentiality, indemnification, payment_terms, data_processing, dispute_resolution, and force_majeure remain unmapped because CUAD has no matching category.
- CUAD has very few title-identifiable NDA, employment, and rental contracts; generic agreement/contract matches are reported separately and are not treated as stronger evidence.
- Remaining problem: title heuristics exclude contracts with no matching title keyword, so the filtered dataset is not all CUAD contracts.
