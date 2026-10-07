# Changes

- Updated dataset diagnostics, document-type matching, and CUAD question visibility.
- Added sliding-window BIO labeling and inference span merging for long documents.
- Added compatible training arguments, metrics, weighted token loss, smoke-test flags,
  and richer training manifests.
- Added stratified split generation safeguards without changing the checked-in split.
- Added train-only reference statistics and expected-clause artifact generation.
- Added configurable expected-clause loading to aggregation.
- Replaced character-array evaluation with exact and IoU-based span metrics and an
  optional TF-IDF baseline.
- Renamed the fixture module to `doc_analysis_fixtures.py` and added development
  requirements and ignore rules.
- Removed synthetic bootstrap fallback and made bootstrap sampling deterministic.

Assumptions: adjacent same-type predictions are one clause when merging windows;
document classification remains limited to the first 512 tokens as specified.
No datasets, APIs, or training were run.
