import collections
import json
import os
import re
from typing import Any
# WARNING: The CUAD dataset contains very long file paths.
# If you run this on Windows, you MUST enable Long Paths in the Windows Registry 
# (Computer\HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled = 1)
# or the Hugging Face dataset download will fail with a FileNotFoundError.
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(os.getcwd(), ".hf_cache"))
os.environ.setdefault("HF_HOME", os.path.join(os.getcwd(), ".hf_cache"))
from datasets import load_dataset
from contract_constants import CLAUSE_TYPES, DOCUMENT_TYPES

def map_cuad_question_to_clause_type(question: str) -> str | None:
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

def determine_document_type(title: str) -> str | None:
    """Uses keyword heuristics to guess document type."""
    t = title.lower()
    if re.search(r"\bnda\b", t) or "non-disclosure" in t or "confidentiality" in t:
        return "nda"
    if re.search(r"\bservice\b", t) or re.search(r"\bmsa\b", t) or "statement of work" in t:
        return "service"
    if re.search(r"\bip\b", t) or "intellectual property" in t or "license" in t or "licensing" in t:
        return "ip"
    return None

def load_and_filter_cuad_dataset(split: str = "train") -> list[dict[str, Any]]:
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


def load_custom_annotations(file_path: str) -> list[dict[str, Any]]:
    """
    Loads custom manual annotations (e.g., from the bootstrap pipeline).
    Expected to be a JSON file with a list of documents matching the CUAD subset shape.
    """
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Custom annotations must be a JSON list")
    validate_documents(data)
    return data


def validate_documents(documents: list[dict[str, Any]]) -> None:
    """Validate the shared document shape before training or splitting."""
    seen: set[str] = set()
    for document in documents:
        name = document.get("document_name")
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError(f"Invalid or duplicate document_name: {name!r}")
        seen.add(name)
        if document.get("document_type") not in DOCUMENT_TYPES:
            raise ValueError(f"Invalid document_type for {name!r}")
        text = document.get("text")
        if not isinstance(text, str):
            raise ValueError(f"Document {name!r} has no text")
        for clause in document.get("clauses", []):
            if clause.get("clause_type") not in CLAUSE_TYPES:
                raise ValueError(f"Invalid clause_type in {name!r}")
            start, end = clause.get("span_start"), clause.get("span_end")
            if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start <= end <= len(text):
                raise ValueError(f"Invalid span in {name!r}")
            if text[start:end] != clause.get("clause_text"):
                raise ValueError(f"Clause span does not match text in {name!r}")

def get_full_dataset(custom_annotations_path="bootstrap_labels_corrected.json", split="train"):
    """
    Returns the combined dataset (CUAD subset + custom annotations) ready for fine-tuning.
    """
    cuad_docs = load_and_filter_cuad_dataset(split)
    custom_docs = load_custom_annotations(custom_annotations_path)
    documents = cuad_docs + custom_docs
    validate_documents(documents)
    return documents


def print_dataset_diagnostics(documents: list[dict[str, Any]]) -> None:
    """Print document and clause coverage, including unused clause labels."""
    document_counts = collections.Counter(
        document["document_type"] for document in documents
    )
    clause_counts = collections.Counter(
        clause["clause_type"]
        for document in documents
        for clause in document.get("clauses", [])
    )
    print("Document types:", dict(document_counts))
    print("Clause types:", dict(clause_counts))
    print(
        "Clause types with zero examples:",
        [clause_type for clause_type in CLAUSE_TYPES if not clause_counts[clause_type]],
    )


def print_cuad_question_examples(split: str = "train", limit: int = 10) -> None:
    """Print distinct raw CUAD questions grouped by their mapped clause type."""
    dataset = load_dataset("TheAtticusProject/cuad", split=split)
    questions: dict[str, set[str]] = collections.defaultdict(set)
    for row in dataset:
        clause_type = map_cuad_question_to_clause_type(row["question"])
        if clause_type:
            questions[clause_type].add(row["question"])
    for clause_type in CLAUSE_TYPES:
        print(f"CUAD questions for {clause_type}:")
        for question in sorted(questions.get(clause_type, set()))[:limit]:
            print(f"  - {question}")

if __name__ == "__main__":
    docs = load_and_filter_cuad_dataset("train")
    print(f"Loaded {len(docs)} documents matching our criteria.")
    
    print_dataset_diagnostics(docs)
    print_cuad_question_examples("train")
