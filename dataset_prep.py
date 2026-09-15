import os
import collections
# WARNING: The CUAD dataset contains very long file paths.
# If you run this on Windows, you MUST enable Long Paths in the Windows Registry 
# (Computer\HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled = 1)
# or the Hugging Face dataset download will fail with a FileNotFoundError.
os.environ["HF_DATASETS_CACHE"] = "C:/hf_cuad"
os.environ["HF_HOME"] = "C:/hf_cuad"
from datasets import load_dataset

def map_cuad_question_to_clause_type(question: str) -> str:
    """Maps CUAD question labels to our specific CLAUSE_TYPES."""
    q = question.lower()
    if "governing law" in q: return "governing_law"
    if "indemnification" in q: return "indemnification"
    if "limitation of liability" in q: return "liability_limitation"
    if "anti-assignment" in q: return "assignment"
    if "non-compete" in q: return "non_compete"
    if "no-solicit" in q: return "non_solicitation"
    if "ip ownership" in q: return "ip_ownership"
    if "renewal term" in q: return "renewal"
    if "force majeure" in q: return "force_majeure"
    if "warranty" in q: return "warranty"
    if "termination for convenience" in q: return "termination"
    return None

def determine_document_type(title: str) -> str:
    """Uses keyword heuristics to guess document type."""
    t = title.lower()
    if "nda" in t or "non-disclosure" in t or "confidentiality" in t:
        return "nda"
    if "service" in t or "msa" in t or "statement of work" in t:
        return "service"
    if "ip " in t or "intellectual property" in t or "license" in t or "licensing" in t:
        return "ip"
    return None

def load_and_filter_cuad_dataset(split="train"):
    """
    Loads CUAD and returns a list of document dicts:
    {
        "document_name": str,
        "document_type": str,
        "text": str,
        "clauses": [
            {
                "clause_text": str,
                "clause_type": str,
                "span_start": int,
                "span_end": int,
            }, ...
        ]
    }
    """
    dataset = load_dataset("TheAtticusProject/cuad", split=split)
    
    # Group by document (title)
    docs_by_title = collections.defaultdict(lambda: {"context": "", "clauses": []})
    
    for row in dataset:
        title = row["title"]
        context = row["context"]
        question = row["question"]
        answers = row["answers"]
        
        # We assume context is the same for the same title
        docs_by_title[title]["context"] = context
        
        clause_type = map_cuad_question_to_clause_type(question)
        if not clause_type:
            continue
            
        # extract clauses
        if len(answers["text"]) > 0:
            for text, start in zip(answers["text"], answers["answer_start"]):
                docs_by_title[title]["clauses"].append({
                    "clause_text": text,
                    "clause_type": clause_type,
                    "span_start": start,
                    "span_end": start + len(text)
                })
                
    # Filter by document type and format
    final_documents = []
    for title, doc_info in docs_by_title.items():
        doc_type = determine_document_type(title)
        if not doc_type:
            continue
            
        final_documents.append({
            "document_name": title,
            "document_type": doc_type,
            "text": doc_info["context"],
            "clauses": doc_info["clauses"]
        })
        
    return final_documents

import json
def load_custom_annotations(file_path: str):
    """
    Loads custom manual annotations (e.g., from the bootstrap pipeline).
    Expected to be a JSON file with a list of documents matching the CUAD subset shape.
    """
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def get_full_dataset(custom_annotations_path="bootstrap_labels_corrected.json", split="train"):
    """
    Returns the combined dataset (CUAD subset + custom annotations) ready for fine-tuning.
    """
    cuad_docs = load_and_filter_cuad_dataset(split)
    custom_docs = load_custom_annotations(custom_annotations_path)
    return cuad_docs + custom_docs

if __name__ == "__main__":
    docs = load_and_filter_cuad_dataset("train")
    print(f"Loaded {len(docs)} documents matching our criteria.")
    
    # Print some stats
    doc_types = collections.Counter(d["document_type"] for d in docs)
    print("Document types:", dict(doc_types))
    
    clause_types = collections.Counter(c["clause_type"] for d in docs for c in d["clauses"])
    print("Clause types:", dict(clause_types))
