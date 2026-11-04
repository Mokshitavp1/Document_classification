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
- Added complete setup, testing, inference, and pipeline documentation to README.
- Added a configurable split output path and fixed the pipeline to honor it.
- Hardened split generation for document types with fewer than four examples.
- Added regression coverage for custom split paths and small-class splitting.
- Cleared the remaining Pylance diagnostic in the weighted trainer.
- Fixed Windows CP1252 pipeline output by using ASCII-safe status messages.
- Fixed artifact generation to use the combined dataset represented by splits.
- Batched sentence-transformer embeddings when building reference statistics.
- Narrowed the evidence-backed report scope to service/IP documents and the nine
  CUAD-supported clause types; retained broader enums for compatibility.
- Added evaluation metadata that records the checkpoint manifest and flags
  evaluations made with a smoke-test-sized training artifact.

Assumptions: adjacent same-type predictions are one clause when merging windows;
document classification remains limited to the first 512 tokens as specified.
No datasets, APIs, or training were run.
