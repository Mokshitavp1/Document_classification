# Legal Document Classification & Clause Extraction — Contracts & Setup

This is the sister document to Devil's Advocate's `CONTRACTS.md`. It governs two
separate things — keep both in mind when reading it:

1. **Internal contracts** — how you and your teammate work on *this* project
   together without blocking each other or silently mismatching shapes.
2. **No-collision rules** — how this project stays mergeable into the Devil's
   Advocate repo later with zero filename/path/key collisions.

If either of you changes a shape, enum, or file ownership boundary here, update
this file in the same commit — that's the whole point of it.

---

## 0. Relationship to Devil's Advocate

- This is a **separate repo**, developed and versioned independently.
- It is **not** wired into Devil's Advocate's `app.py`, `CONTRACTS.md`, or pipeline
  at any point during research/training. No shared imports, no shared data paths,
  no shared session state, until an explicit merge decision is made.
- When (if) this proves out, the plan is to copy the finished module wholesale into
  the Devil's Advocate repo as a new top-level folder — see Section 8.

---

## 1. Team split & ownership

Mirrors the Person A / Person B pattern from Devil's Advocate — same reasoning:
each person can build and test their half using fixture data (Section 5), with
zero dependency on the other's code being finished.

- **Person A — data & modeling.** Owns `dataset_prep.py`, `train.py`,
  `evaluate.py`, `legal_bert_model.py`. Responsible for: sourcing/annotating
  data, the train/val/test split (Section 7), fine-tuning, and producing the two
  inference functions everything downstream depends on:
  - `classify_document(text) -> <3.1 shape>`
  - `extract_clauses(text) -> list[<3.2 shape>]`
- **Person B — application & integration.** Owns `clause_extractor.py` (the
  unusual-clause flagging logic, i.e. filling in `flagged_unusual` from 3.2) and
  `aggregate.py` (building the 3.3 document-level result), plus any UI/demo code.
  Consumes Person A's function *outputs only* — never imports
  `legal_bert_model.py` internals directly.
- **If a function needs a field or shape not listed in Section 4, don't guess —
  add it here first**, same rule as Devil's Advocate's own contract doc.

Reassign names once you actually decide who does what — keep the boundary and the
fixture-decoupling pattern regardless of who's "A" and who's "B."

---

## 2. Environment — install separately from Devil's Advocate

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Expected additions on top of what Devil's Advocate already uses:
```
transformers>=4.40.0
torch>=2.2.0
datasets>=2.19.0
scikit-learn>=1.4.0
seqeval>=1.2.2        # NER / clause-span evaluation
accelerate>=0.30.0    # only if fine-tuning, not needed for inference-only
```

**Model checkpoint**, pulled from Hugging Face, not Ollama — this project does not
use `qwen2.5:7b-instruct` at all. **Decision: `nlpaueb/legal-bert-base-uncased`.**
It's the most established legal-domain BERT checkpoint (pretrained on EU/UK/US
legislation, court cases, and contracts), 110M params — comfortably runs and
fine-tunes on a laptop, consistent with staying on-device. Whoever pulls it first,
record the exact revision hash here immediately so the other person pins the same
one:
```bash
# nlpaueb/legal-bert-base-uncased, revision 15b570cbf88259610b082a167dacc190124f60f6
```

**Environment parity checklist (do this explicitly, don't assume):**
- [ ] Both machines run `pip freeze | grep -E "torch|transformers|datasets"` and
      diff the output — version mismatches here are the most common cause of
      "works on my machine" bugs with HF models
- [ ] Both machines confirm the identical checkpoint + revision hash (not just
      the model name — checkpoints get updated upstream)
- [ ] `requirements.txt` is identical on both machines (`pip freeze` and diff if
      unsure), same as Devil's Advocate's own checklist

**Dependency-conflict check (do this before merge, not after):** diff this
`requirements.txt` against Devil's Advocate's. `torch`/`transformers` are the
likely collision point if Devil's Advocate ever adds its own transformer usage —
confirm versions are compatible, don't just concatenate the two files blindly.

---

## 3. No-Collision Rules (for the eventual merge)

Devil's Advocate already owns these names/paths. **Do not reuse any of them**,
even for something conceptually similar — pick a different name so a future merge
is a pure file-copy, not a rename-and-pray:

| Category | Already used by Devil's Advocate — AVOID | Use instead, in this project |
|---|---|---|
| Filenames | `ingest.py`, `parser.py`, `summarizer.py`, `generate.py`, `verifier.py`, `hyde.py`, `self_rag.py`, `stage1_case_retrieval.py`, `stage2_chunk_retrieval.py`, `router.py`, `app.py`, `theme.py`, `fixtures.py`, `config.toml` | `classify.py` (n/a — see Section 1, use `legal_bert_model.py`), `clause_extractor.py`, `dataset_prep.py`, `train.py`, `evaluate.py`, `aggregate.py` |
| Vector store path | `data/chroma_db` | `doc_analysis_data/chroma_db` (only if this project ends up needing its own store at all — see Section 6) |
| Chroma collections | `legal_chunks`, `legal_cases` | `contract_clauses`, `contract_documents` |
| Streamlit session-state keys | `selected_mode`, `query_draft`, `pending_warning`, `run_after_warning`, `last_query_result` | prefix everything: `doc_analysis_selected_doc`, `doc_analysis_result`, etc. |
| Streamlit widget `key=` values | `case-pdf-upload`, `build-knowledge-base`, `legal-query`, `mode-fast`/`mode-deep`/`mode-auto`, `switch-mode`, `continue-mode` | prefix everything: `doc-analysis-upload`, `doc-analysis-classify-button`, etc. |
| CSS custom properties | `--mist`, `--sand`, `--tan`, `--brown`, `--canvas`, `--paper`, `--muted`, `--line` (from `theme.py`) | reuse the same palette *values* for visual consistency if this ever gets its own UI, but import Devil's Advocate's `theme.py` at merge time rather than maintaining a parallel one now |
| Config/env constant names | `CHROMA_PATH`, `CHUNKS_COLLECTION`, `CASES_COLLECTION`, `EMBEDDING_MODEL_NAME` | `CLAUSE_MODEL_NAME`, `CLASSIFIER_CHECKPOINT_PATH` — different constant names even where the *shape* of the constant is similar |

If a name isn't in the table and you're unsure, check the actual files in the
Devil's Advocate repo before claiming it — this table reflects the codebase as of
this doc's writing and may drift.

---

## 4. Data contracts — internal to this project

Field names are deliberately **not** reused from Devil's Advocate's contracts
(`relevance_score`, `case_name`, `page_number`) even where the concept overlaps,
so nothing downstream can confuse a clause-classification result for a
case-retrieval result if the two pipelines ever run in the same process.

### 4.1 Classification result — one per uploaded document (from Person A's `classify_document`)
```python
{
    "document_name": str,      # uploaded filename without extension
    "document_type": str,      # MUST be one of DOCUMENT_TYPES (see 4.3) — no free text
    "confidence_score": float, # 0.0–1.0, higher = more confident
}
```

### 4.2 Extracted clause — one per identified clause (from Person A's `extract_clauses`)
```python
{
    "clause_text": str,
    "clause_type": str,        # MUST be one of CLAUSE_TYPES (see 4.3) — no free text
    "document_name": str,
    "span_start": int,         # character offset in source document
    "span_end": int,
    "confidence_score": float, # 0.0–1.0
}
```
Note: `flagged_unusual` is deliberately NOT produced here — that's Person B's
`clause_extractor.py` logic, applied as a post-processing step (4.4). Keeping it
out of Person A's raw model output means the "what counts as unusual" definition
can change without retraining anything.

### 4.3 Fixed enums — the actual authoritative lists, not `e.g.` examples

```python
DOCUMENT_TYPES = ["employment", "rental", "privacy", "ip", "service", "nda"]

CLAUSE_TYPES = [
    "termination", "confidentiality", "indemnification", "governing_law",
    "payment_terms", "liability_limitation", "assignment", "non_compete",
    "non_solicitation", "ip_ownership", "data_processing", "renewal",
    "dispute_resolution", "force_majeure", "warranty",
]
```
**Decision: this is the working list, effective now** — not every category applies
to every document type (e.g. `non_compete` won't show up in a privacy policy), and
that's fine; a clause type simply won't appear for document types it doesn't apply
to. Whoever labels data owns keeping this list in sync with the actual label set
used — if annotation surfaces a clause type genuinely missing here (or one of
these never occurs in real data), update the list in this file first, then in
code. Don't let the two drift.

### 4.4 Clause with unusual-flag applied (from Person B's `clause_extractor.py`)
```python
{
    **{ <4.2 shape> },
    "flagged_unusual": bool,   # added by Person B's post-processing, not the model
    "flag_reason": str | None, # short human-readable reason if flagged, else None
}
```

**Decision — "unusual" is defined two ways, and `flag_reason` says which one fired:**

1. **Outlier clause (clause-level):** embed `clause_text` (reuse
   `all-MiniLM-L6-v2` for consistency with Devil's Advocate rather than
   introducing a third embedding model), compute cosine similarity to the
   centroid embedding of all *other* clauses sharing the same `clause_type` in
   the training corpus. Flag if similarity falls below the 10th percentile for
   that `clause_type`. `flag_reason`: `"atypical wording for a {clause_type} clause"`.
2. **Missing expected clause (document-level, surfaced per-document not per-clause):**
   maintain a small `EXPECTED_CLAUSES` mapping of `document_type -> set of
   clause_types normally present` (e.g. an NDA missing any `confidentiality`
   clause is itself the signal). This doesn't produce a 4.2 clause dict — surface
   it as a separate field on the 4.5 document-level result instead:
   `"missing_expected_clauses": list[str]`.
   Person B builds `EXPECTED_CLAUSES` from the training corpus (which clause
   types actually co-occur with which document types), not by guessing.

Both are cheap, need no extra model, and are explainable in a demo — which
matters more here than a fancier anomaly-detection approach would.

### 4.5 Document-level result (from Person B's `aggregate.py` — what a consumer/UI receives)
```python
{
    "document_name": str,
    "classification": { <4.1 shape, minus document_name> },
    "clauses": [ <4.4 shape>, ... ],
    "missing_expected_clauses": [str, ...],  # clause_types expected for this
                                              # document_type but not found — see 4.4
}
```

---

## 5. Working in parallel — fixtures, so neither of you blocks the other

Person B's `clause_extractor.py` and `aggregate.py` take plain dicts as input —
never a live model or checkpoint. That means Person B can build and test their
entire half using fixture data (create a `fixtures.py` here, same idea as Devil's
Advocate's), with zero dependency on Person A's model being trained or even
started.

```python
# fixtures.py — example shape, fill in with realistic sample text once you have it
SAMPLE_CLASSIFICATION = {
    "document_name": "sample_nda_001",
    "document_type": "nda",
    "confidence_score": 0.91,
}

SAMPLE_CLAUSES = [
    {
        "clause_text": "This Agreement shall remain in effect for a period of...",
        "clause_type": "termination",
        "document_name": "sample_nda_001",
        "span_start": 512,
        "span_end": 610,
        "confidence_score": 0.88,
    },
]
```

Only at integration time does Person B swap fixture calls for Person A's real
`legal_bert_model.py` functions.

---

## 6. Do you even need a vector store for this project?

Worth deciding explicitly rather than defaulting to "add ChromaDB because Devil's
Advocate has one." Classification + clause extraction is a supervised
inference/training task, not a retrieval task — you likely only need:
- Raw document storage (filesystem is fine for research)
- Model checkpoints (HF cache or local dir, gitignored)
- A labeled dataset (see open questions)

Only add a vector store if you end up doing embedding-based semantic-similarity
comparisons (the "compare general-purpose and legal-domain representations" part
of the research extension) — and if so, use the namespacing in Section 3, not
Devil's Advocate's paths, even in early experiments.

---

## 6a. Dataset strategy — decision

CUAD covers `service`, `ip`, and `nda`-adjacent clauses reasonably well (it's
built from commercial/licensing contracts); it does not meaningfully cover
`employment`, `rental`, or `privacy`. Splitting the approach by category, rather
than picking one strategy for all six, is the honest move:

- **`service`, `ip`, `nda`:** fine-tune on the relevant CUAD subset. Cite CUAD's
  actual coverage in the paper rather than implying full coverage of all six
  categories.
- **`employment`, `rental`, `privacy`:** hand-annotate a small custom set (aim for
  enough per category to fine-tune meaningfully, even if modest — 30-50 documents
  per category is a realistic scope for a two-person team, not thousands).
  Bootstrap the labeling with zero-shot LLM pre-labeling (feed candidate clauses
  to a general-purpose LLM prompted with `CLAUSE_TYPES`, then have a human
  correct it) rather than labeling from scratch by hand — faster, and it's the
  same "generate then verify" pattern Devil's Advocate already uses elsewhere.

This split *is* a real research contribution, not a workaround: "general-domain
CUAD transfer vs. small hand-labeled legal-BERT fine-tune, per document category"
is a legitimate empirical comparison for the paper, and it's more honest than
claiming uniform coverage you don't have.

---

## 7. Reproducibility & evaluation protocol

Agree on these BEFORE either of you trains anything — otherwise your results
aren't comparable and you'll waste time re-running experiments to make them so.

- **Random seed:** fix one (e.g. `42`) and use it everywhere — split, shuffling,
  model init.
- **Split:** fix a ratio (e.g. 70/15/15 train/val/test) and, more importantly,
  fix the *actual assignment* — check a `splits.json` (list of document IDs per
  split) into the repo so both of you train/evaluate on identically the same
  documents, regardless of who runs it.
- **Checkpoint naming:** `checkpoints/legal_bert_clause_v{n}/`, incrementing `n`
  per meaningfully different training run — don't overwrite `v1` in place.
- **Evaluation metric:** macro-F1 for `classify_document` (document type is a
  multi-class problem); span/token-level F1 via `seqeval` for `extract_clauses`.
  Report both on the held-out test set only — no peeking during development.
- **What counts as "done":** agree on this number now, even roughly, so neither
  of you is guessing later whether the model is good enough to move on from.

---

## 8. Git workflow

- **Never commit:** raw training data/annotations if under any redistribution
  restriction, model checkpoints (use HF cache or a documented download step,
  not `git add` on multi-hundred-MB files), `venv/`, `__pycache__/`.
- Add to `.gitignore`:
  ```
  data/
  checkpoints/
  venv/
  __pycache__/
  *.bin
  *.safetensors
  ```
- If the fine-tuned model becomes the actual deliverable, decide now whether it
  ships via Git LFS, a Hugging Face Hub upload, or a documented manual download —
  don't let this be an afterthought at merge time.

---

## 9. Merge checklist (do this together, once, if/when this gets folded into Devil's Advocate)

- [ ] Diff this `requirements.txt` against Devil's Advocate's; resolve any version
      conflicts (torch/transformers most likely)
- [ ] Confirm zero filename collisions against Section 3's table (re-check the
      table against the live repo, not just this doc, in case either side drifted)
- [ ] Copy this project in as a single new top-level folder, e.g.
      `document_analysis/` — do not scatter its files across existing folders
- [ ] Add a new section to Devil's Advocate's `CONTRACTS.md` documenting the
      4.1–4.5 shapes above, exactly as this doc defines them
- [ ] Decide UI integration point: new Streamlit tab/mode, not woven into the
      existing case-law query flow (`app.py`'s Fast/Deep Thinking/Auto modes stay
      untouched)
- [ ] Confirm model checkpoint distribution works on the actual demo machine
      (same "must exist locally before demo" rule as Devil's Advocate's
      `data/chroma_db`)
- [ ] Finalize ownership: does this fold under Tina's existing
      `generate.py`/`verifier.py`/UI ownership, or does Person A (Devil's
      Advocate's) take it given its proximity to ingestion?
- [ ] Run one full document through the merged pipeline end-to-end on the demo
      machine before presenting

---

## Remaining open item

- **Exact revision hash for `nlpaueb/legal-bert-base-uncased`** — record it in
  Section 2 the moment either of you pulls the checkpoint, so you're both pinned
  to the same one. Everything else that was previously open (dataset strategy,
  `CLAUSE_TYPES`, the unusual-clause definition) is now decided above — treat
  those as working defaults, and only revise them once real data actually
  contradicts an assumption, not preemptively.
