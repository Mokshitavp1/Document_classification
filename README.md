# Legal Document Classification and Clause Extraction

This project fine-tunes Legal-BERT to experiment with:

1. classify service and IP-related contracts; and
2. extract nine CUAD-supported clause types using BIO token classification:
   `termination`, `governing_law`, `liability_limitation`, `assignment`,
   `non_compete`, `non_solicitation`, `ip_ownership`, `renewal`, and `warranty`.

The code retains the original six-type/fifteen-label enum for compatibility with
the data contracts and future reviewed annotations. The evidence-backed report
scope is narrower because the available CUAD title-matched data contains 429
service documents, 50 IP documents, and only one document each for employment,
rental, privacy, and NDA.

The public inference API is in [`legal_bert_model.py`](legal_bert_model.py), and
the document-level response assembler is in
[`aggregate.py`](aggregate.py).

## Status

The software pipeline is runnable and reproducible. The checked-in checkpoint
is a smoke-test/training artifact trained for one epoch on 30 documents, not a
claim of production accuracy. Its first three training-document probes produced
zero predicted clauses, so its clause score is a training failure signal rather
than evidence that offset evaluation is correct. The latest recorded
evaluation reports classification macro-F1 `0.471` versus a TF-IDF baseline of
`0.571`, and clause-span F1 `0.0`.
See [`DATASET_REPORT.md`](DATASET_REPORT.md) and
[`evaluation_results.json`](evaluation_results.json) before using the model.

## Setup

PowerShell:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

The training and evaluation scripts download the pinned
`nlpaueb/legal-bert-base-uncased` revision from Hugging Face. Network access is
required on the first run. Raw data and model checkpoints are intentionally
ignored by git.

## Test and syntax checks

```powershell
pytest -q
python -m compileall -q .
```

## Reproducible pipeline

The full pipeline generates a fixed split, trains both models, builds
training-only reference artifacts, and evaluates the held-out test set:

```powershell
python run_pipeline.py --epochs 3 --baseline tfidf
```

For a reportable run, omit `--max-train-docs` and use the full 339-document
training split. GPU execution on Colab or Kaggle is recommended. Evaluation
records the training manifest and marks results as `is_smoke_test` when the
checkpoint was trained on fewer documents than the configured split.

Useful options:

```text
--splits PATH          Split manifest (default: splits.json)
--output PATH          Checkpoint directory
--epochs N             Number of training epochs
--max-train-docs N     Limit training documents for a smoke run
--skip-train           Reuse an existing checkpoint
--baseline tfidf       Include the TF-IDF comparison
```

For a quick local smoke run:

```powershell
python run_pipeline.py --max-train-docs 8 --epochs 1 --output checkpoints/smoke
```

Do not overwrite an established `splits.json` unless intentionally changing
the reproducibility contract. A custom split location can be generated with:

```powershell
python generate_splits.py --output path/to/splits.json
```

## Inference

Set `CLASSIFIER_CHECKPOINT_PATH` before importing the inference module:

```powershell
$env:CLASSIFIER_CHECKPOINT_PATH = "checkpoints/legal_bert_clause_v1"
python -c "from legal_bert_model import classify_document; print(classify_document('This service agreement renews annually.'))"
```

`aggregate_document` can load `expected_clauses.json` from the configured
checkpoint automatically. It returns the classification, extracted clauses,
and any expected clause types missing from the result.

## Data and artifacts

- `bootstrap_labels_corrected.json` is the reviewed custom annotation input.
- `bootstrap_labels_pending_review.json` is not training data and remains
  ignored until manually reviewed.
- `splits.json` fixes the train/validation/test assignment.
- `training_manifest.json` records the base-model revision, seed, versions, and
  split counts.
- `reference_stats.json` and `expected_clauses.json` are generated from the
  training split only.

CUAD provides useful coverage for service and IP-related agreements but does
not provide balanced coverage for all six document types or all fifteen clause
types. The pending bootstrap annotations are not included because they have not
been reviewed. Add reviewed annotations, train on the full split, probe training
documents for non-empty predictions, and regenerate the dataset report before
making broader accuracy claims.

## Project contracts

Input/output shapes, fixed enums, reproducibility rules, and integration
requirements are defined in [`CONTRACTS.md`](CONTRACTS.md).
