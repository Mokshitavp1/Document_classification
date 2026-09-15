import os
import json
import random
from datasets import load_dataset

try:
    from openai import OpenAI
except ImportError:
    print("Please install openai: pip install openai")
    exit(1)

CLAUSE_TYPES = [
    "termination", "confidentiality", "indemnification", "governing_law",
    "payment_terms", "liability_limitation", "assignment", "non_compete",
    "non_solicitation", "ip_ownership", "data_processing", "renewal",
    "dispute_resolution", "force_majeure", "warranty"
]

def get_llm_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set it to run the LLM pre-labeling.")
        exit(1)
    return OpenAI(api_key=api_key)

def extract_clauses_with_llm(client, text, document_type):
    prompt = f"""
You are an expert legal AI.
Given the following {document_type} contract, extract any clauses that match the following types:
{', '.join(CLAUSE_TYPES)}

Return the extracted clauses as a JSON list of objects, where each object has:
- "clause_type": string (must be exactly one of the types above)
- "clause_text": string (the exact verbatim text of the clause from the document)

If there are no matching clauses, return an empty list [].
Do NOT include any markdown formatting, return ONLY valid JSON.

Contract Text:
{text[:4000]}  # Truncated for token limits in this script
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
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

def main():
    client = get_llm_client()
    
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

    # Fallback if datasets are completely empty/failed
    if not documents:
        print("Warning: Could not source documents from HF. Creating dummy ones for the pipeline test.")
        documents.append({
            "document_name": "dummy_employment_001",
            "document_type": "employment",
            "text": "This Employment Agreement is entered into today. The employee shall not compete with the company for 1 year after termination."
        })

    print(f"Total documents to process: {len(documents)}")
    
    # Process with LLM
    final_output = []
    for doc in documents:
        print(f"Processing {doc['document_name']}...")
        clauses = extract_clauses_with_llm(client, doc["text"], doc["document_type"])
        final_output.append({
            "document_name": doc["document_name"],
            "document_type": doc["document_type"],
            "text": doc["text"],
            "clauses": clauses
        })
        
    output_file = "bootstrap_labels_pending_review.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
        
    print(f"Done! Saved to {output_file}.")
    print("Please manually review and correct this file, then save it as 'bootstrap_labels_corrected.json'.")

if __name__ == "__main__":
    main()
