import os
import json
import random
from datasets import load_dataset

try:
    from openai import OpenAI
except ImportError:
    print("Please install openai: pip install openai")
    exit(1)

from contract_constants import CLAUSE_TYPES
from dataset_prep import validate_documents

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
            f"\n❌  Cannot reach Ollama at {OLLAMA_HOST}.\n"
            "    Make sure Ollama is running: https://ollama.com\n"
            f"    Error: {exc}"
        )
        exit(1)
    if MODEL_NAME not in models:
        print(
            f"\n❌  Model '{MODEL_NAME}' is not pulled yet.\n"
            f"    Run:  ollama pull {MODEL_NAME}\n"
            f"    Available models: {models or '(none)'}"
        )
        exit(1)
    print(f"✅  Ollama ready — using model '{MODEL_NAME}' at {OLLAMA_HOST}")

def extract_clauses_with_llm(client: OpenAI, text: str, document_type: str) -> list[dict]:
    """Ask the local Qwen2.5-7B model to extract clause spans from a document."""
    truncated_text = text[:_CONTEXT_CHARS]
    prompt = f"""You are an expert legal AI.
Given the following {document_type} contract, extract any clauses that match the following types:
{', '.join(CLAUSE_TYPES)}

Return the extracted clauses as a JSON list of objects, where each object has:
- "clause_type": string (must be exactly one of the types above)
- "clause_text": string (the exact verbatim text of the clause from the document)

If there are no matching clauses, return an empty list [].
Do NOT include any markdown formatting, return ONLY valid JSON.

Contract Text:
{truncated_text}"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = response.choices[0].message.content.strip()
        # Clean up possible markdown fences
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
            
        clauses = json.loads(content)
        
        # Now find the exact spans
        valid_clauses = []
        for c in clauses:
            c_text = c.get("clause_text", "")
            c_type = c.get("clause_type", "")
            
            # Simple string matching to find the offset
            start_idx = text.find(c_text)
            if start_idx != -1 and c_type in CLAUSE_TYPES:
                valid_clauses.append({
                    "clause_text": c_text,
                    "clause_type": c_type,
                    "span_start": start_idx,
                    "span_end": start_idx + len(c_text)
                })
        return valid_clauses
    except Exception as e:
        print(f"Error during LLM extraction: {e}")
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
        # Example dataset for various legal contracts (contains employment and rental)
        print("Downloading general contracts...")
        contracts_ds = load_dataset("albertvillanova/legal_contracts", split="train")
        
        # Filter for employment and rental heuristically
        employment_docs = []
        rental_docs = []
        
        for row in contracts_ds:
            text = row.get("text", "").lower()
            if "employment" in text or "employee" in text:
                employment_docs.append(row)
            elif "rental" in text or "lease" in text or "tenant" in text:
                rental_docs.append(row)
                
            if len(employment_docs) >= 30 and len(rental_docs) >= 30:
                break
                
        for i, sample in enumerate(employment_docs[:30]):
            documents.append({
                "document_name": f"employment_contract_{i:03d}",
                "document_type": "employment",
                "text": sample.get("text", "")
            })
            
        for i, sample in enumerate(rental_docs[:30]):
            documents.append({
                "document_name": f"rental_agreement_{i:03d}",
                "document_type": "rental",
                "text": sample.get("text", "")
            })
            
    except Exception as e:
        print(f"Failed to load legal_contracts dataset: {e}")

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
