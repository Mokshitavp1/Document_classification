import os
import json
import random

# Suppress the HuggingFace symlink warning on Windows — degraded caching is
# acceptable here; we're only downloading small bootstrap datasets.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from datasets import load_dataset

try:
    from openai import OpenAI
except ImportError:
    print("Please install openai: pip install openai")
    exit(1)

from contract_constants import CLAUSE_TYPES
from dataset_prep import validate_documents, CUAD_JSON_URL

# ---------------------------------------------------------------------------
# Model configuration — local Ollama, no API key required.
# Override OLLAMA_HOST if Ollama is not running on localhost.
# ---------------------------------------------------------------------------
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = "qwen2.5:7b-instruct"
# Qwen2.5-7B has a 32 k context window; 6 000 chars is a safe truncation point
# that keeps us well within limits even after the system prompt overhead.
_CONTEXT_CHARS = 6_000

def get_llm_client() -> OpenAI:
    """Return an OpenAI-compatible client pointed at the local Ollama server."""
    return OpenAI(
        base_url=f"{OLLAMA_HOST}/v1",
        # Ollama's OpenAI-compatible endpoint accepts any non-empty string.
        api_key="ollama",
    )


def check_ollama_ready(client: OpenAI) -> None:
    """Verify Ollama is running and the required model is available."""
    try:
        models = [m.id for m in client.models.list().data]
    except Exception as exc:
        print(
            f"\n[ERROR] Cannot reach Ollama at {OLLAMA_HOST}.\n"
            "        Make sure Ollama is running: https://ollama.com\n"
            f"        Error: {exc}"
        )
        exit(1)
    if MODEL_NAME not in models:
        print(
            f"\n[ERROR] Model '{MODEL_NAME}' is not pulled yet.\n"
            f"        Run:  ollama pull {MODEL_NAME}\n"
            f"        Available models: {models or '(none)'}"
        )
        exit(1)
    print(f"[OK] Ollama ready - using model '{MODEL_NAME}' at {OLLAMA_HOST}")

def _parse_llm_json(content: str) -> list[dict] | None:
    """Extract a JSON list from the model response using three fallback strategies.

    Returns the parsed list on success, or None if all strategies fail.
    """
    import re as _re

    # Strategy 1: strip markdown fences, then parse directly.
    text = content.strip()
    for fence in ("```json", "```"):
        if text.startswith(fence):
            text = text[len(fence):]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            break
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Strategy 2: pull the first [...] block out even if there's surrounding prose.
    match = _re.search(r"\[.*?\]", text, _re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Strategy 3: truncate at the last complete object to repair trailing garbage.
    last_brace = text.rfind("}")
    if last_brace != -1:
        candidate = text[: last_brace + 1]
        # Wrap as array if not already one.
        if not candidate.lstrip().startswith("["):
            candidate = "[" + candidate + "]"
        else:
            candidate = candidate + "]"
        try:
            result = json.loads(candidate)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    return None


def extract_clauses_with_llm(client: OpenAI, text: str, document_type: str) -> list[dict]:
    """Ask the local Qwen2.5-7B model to extract clause spans from a document.

    Retries up to 3 times on JSON parse failure before returning an empty list.
    """
    truncated_text = text[:_CONTEXT_CHARS]
    base_prompt = (
        f"You are an expert legal AI.\n"
        f"Given the following {document_type} contract, extract any clauses that "
        f"match the following types:\n{', '.join(CLAUSE_TYPES)}\n\n"
        f"Return the extracted clauses as a JSON list of objects, where each object has:\n"
        f'- "clause_type": string (must be exactly one of the types above)\n'
        f'- "clause_text": string (the exact verbatim text of the clause from the document)\n\n'
        f"If there are no matching clauses, return an empty list [].\n"
        f"Do NOT include any markdown formatting, return ONLY valid JSON.\n\n"
        f"Contract Text:\n{truncated_text}"
    )

    _MAX_ATTEMPTS = 3
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        # On retries, prepend an explicit reminder to fix the format.
        retry_prefix = (
            f"Your previous response was not valid JSON. "
            f"Return ONLY a JSON array, nothing else. Attempt {attempt}/{_MAX_ATTEMPTS}.\n\n"
            if attempt > 1 else ""
        )
        prompt = retry_prefix + base_prompt
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            raw = response.choices[0].message.content
            parsed = _parse_llm_json(raw)
            if parsed is None:
                raise json.JSONDecodeError(
                    f"Could not extract a JSON list from response", raw, 0
                )
            # Resolve character spans in the original (un-truncated) text.
            valid_clauses = []
            for c in parsed:
                c_text = c.get("clause_text", "")
                c_type = c.get("clause_type", "")
                start_idx = text.find(c_text)
                if start_idx != -1 and c_type in CLAUSE_TYPES:
                    valid_clauses.append({
                        "clause_text": c_text,
                        "clause_type": c_type,
                        "span_start": start_idx,
                        "span_end": start_idx + len(c_text),
                    })
            return valid_clauses
        except Exception as exc:
            if attempt < _MAX_ATTEMPTS:
                print(f"  Parse error (attempt {attempt}/{_MAX_ATTEMPTS}), retrying: {exc}")
            else:
                print(f"  Failed after {_MAX_ATTEMPTS} attempts: {exc}")
    return []


def main() -> None:
    random.seed(42)
    client = get_llm_client()
    check_ollama_ready(client)
    
    print("Sourcing documents from Hugging Face...")
    
    documents = []
    
    # We use a try-catch for dataset loading to gracefully handle any HF connection issues
    try:
        # Example dataset for privacy policies
        print("Downloading privacy policies...")
        privacy_ds = load_dataset("sjsq/PrivacyPolicy", split="train")
        privacy_samples = random.sample(list(privacy_ds), min(30, len(privacy_ds)))
        for i, sample in enumerate(privacy_samples):
            documents.append({
                "document_name": f"privacy_policy_{i:03d}",
                "document_type": "privacy",
                "text": sample.get("text", "")
            })
    except Exception as e:
        print(f"Failed to load privacy dataset: {e}")
        
    try:
        # albertvillanova/legal_contracts permanently broke in datasets 3.x
        # (custom loading scripts removed). We now pull employment and rental
        # examples directly from CUAD — the same dataset dataset_prep.py uses —
        # filtering by text keywords rather than title only, so we catch more
        # contracts than the strict title heuristic in dataset_prep.py.
        print("Downloading employment and rental contracts from CUAD...")
        cuad_raw = load_dataset(
            "json",
            data_files={"cuad": CUAD_JSON_URL},
            field="data",
        )

        employment_docs: list[dict] = []
        rental_docs: list[dict] = []

        splits = cuad_raw.values() if hasattr(cuad_raw, "values") else (cuad_raw,)
        for split_ds in splits:
            for article in split_ds:
                title = article.get("title", "")
                paragraphs = article.get("paragraphs") or []
                context = paragraphs[0]["context"] if paragraphs else ""
                # Probe the title + first 500 chars of body for keywords.
                probe = (title + " " + context[:500]).lower()

                if len(employment_docs) < 30 and (
                    "employment" in probe
                    or "employee" in probe
                    or "employer" in probe
                ):
                    employment_docs.append({"text": context, "title": title})
                elif len(rental_docs) < 30 and (
                    "lease" in probe
                    or "rental" in probe
                    or "tenant" in probe
                    or "landlord" in probe
                ):
                    rental_docs.append({"text": context, "title": title})

                if len(employment_docs) >= 30 and len(rental_docs) >= 30:
                    break

        for i, sample in enumerate(employment_docs):
            documents.append({
                "document_name": f"employment_contract_{i:03d}",
                "document_type": "employment",
                "text": sample["text"],
            })

        for i, sample in enumerate(rental_docs):
            documents.append({
                "document_name": f"rental_agreement_{i:03d}",
                "document_type": "rental",
                "text": sample["text"],
            })

        print(
            f"  Found {len(employment_docs)} employment and "
            f"{len(rental_docs)} rental documents from CUAD."
        )

    except Exception as e:
        print(f"Failed to load employment/rental from CUAD: {e}")

    if not documents:
        raise RuntimeError(
            "No source documents were loaded; refusing to create synthetic bootstrap data."
        )

    print(f"Total documents to process: {len(documents)}")
    
    # Process with LLM
    final_output = []
    for doc in documents:
        print(f"Processing {doc['document_name']}...")
        # The bootstrap LLM sees only the first 4000 characters of each document.
        clauses = extract_clauses_with_llm(client, doc["text"], doc["document_type"])
        final_output.append({
            "document_name": doc["document_name"],
            "document_type": doc["document_type"],
            "text": doc["text"],
            "clauses": clauses
        })
        
    try:
        validate_documents(final_output)
    except ValueError as exc:
        print(f"Warning: bootstrap output failed validation: {exc}")
        print("Saving anyway for manual review, but fix the issues before using for training.")

    output_file = "bootstrap_labels_pending_review.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"Done! Saved to {output_file}.")
    print("Please manually review and correct this file, then save it as 'bootstrap_labels_corrected.json'.")

if __name__ == "__main__":
    main()
