import collections
import json
import os
import random
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


CUAD_JSON_URL = (
    "https://huggingface.co/datasets/theatticusproject/cuad/"
    "resolve/main/CUAD_v1/CUAD_v1.json"
)


def map_cuad_question_to_clause_type(question: str) -> str | None:
    """Maps CUAD question labels to our specific CLAUSE_TYPES."""
    match = re.search(r'related to "([^"]+)"', question, flags=re.IGNORECASE)
    category = match.group(1).casefold() if match else question.casefold()
    mappings = {
        "anti-assignment": "assignment",
        "cap on liability": "liability_limitation",
        "governing law": "governing_law",
        "ip ownership assignment": "ip_ownership",
        "no-solicit of customers": "non_solicitation",
        "no-solicit of employees": "non_solicitation",
        "non-compete": "non_compete",
        "renewal term": "renewal",
        "termination for convenience": "termination",
        "uncapped liability": "liability_limitation",
        "warranty duration": "warranty",
    }
    return mappings.get(category)

def determine_document_type(title: str) -> str | None:
    """Uses keyword heuristics to guess document type."""
    match = _document_type_match(title)
    return match[0] if match else None


_DOCUMENT_TYPE_PATTERNS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "nda",
        "specific",
        (r"nda", r"non[- ]disclosure", r"confidentiality"),
    ),
    (
        "privacy",
        "specific",
        (r"privacy", r"data protection", r"data privacy"),
    ),
    (
        "employment",
        "specific",
        (r"employment", r"employee", r"executive", r"offer letter"),
    ),
    (
        "rental",
        "specific",
        (r"lease", r"rental", r"rent", r"sublease", r"real estate"),
    ),
    (
        "ip",
        "specific",
        (
            r"intellectual property",
            r"\bip\b",
            r"license",
            r"licensing",
            r"patent",
            r"trademark",
            r"copyright",
            r"technology",
        ),
    ),
    (
        "service",
        "specific",
        (
            r"service",
            r"msa",
            r"statement of work",
            r"affiliate",
            r"agency",
            r"collaboration",
            r"co-branding",
            r"consulting",
            r"development",
            r"distributor",
            r"endorsement",
            r"franchise",
            r"hosting",
            r"joint venture",
            r"maintenance",
            r"manufacturing",
            r"marketing",
            r"outsourcing",
            r"promotion",
            r"reseller",
            r"sponsorship",
            r"strategic alliance",
            r"supply",
            r"transportation",
        ),
    ),
    ("service", "generic", (r"agreement", r"contract")),
)


def _document_type_match(title: str) -> tuple[str, str] | None:
    """Return the type and whether its match is specific or generic."""
    normalized_title = title.casefold()
    for document_type, match_kind, patterns in _DOCUMENT_TYPE_PATTERNS:
        pattern = r"(?<!\w)(?:" + "|".join(patterns) + r")(?!\w)"
        if re.search(pattern, normalized_title):
            return document_type, match_kind
    return None


def _cuad_title_classification_counts(
    articles: list[dict[str, Any]],
) -> collections.Counter[str]:
    """Count CUAD titles by specific, generic-only, or unmatched classification."""
    counts: collections.Counter[str] = collections.Counter()
    for article in articles:
        match = _document_type_match(article["title"].strip())
        if match is None:
            counts["unmatched"] += 1
        elif match[1] == "generic":
            counts["generic_only"] += 1
        else:
            counts[match[0]] += 1
    return counts


def _load_cuad_articles() -> list[dict[str, Any]]:
    """Load every contract from the official, script-free CUAD JSON export."""
    dataset = load_dataset(
        "json",
        data_files={"cuad": CUAD_JSON_URL},
        field="data",
    )
    articles: list[dict[str, Any]] = []
    datasets_by_split = dataset.values() if hasattr(dataset, "values") else (dataset,)
    for split_dataset in datasets_by_split:
        articles.extend(split_dataset)
    return articles


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
    del split  # The official JSON export contains the train/test contracts together.
    dataset = _load_cuad_articles()

    final_documents = []
    used_names: set[str] = set()
    for article_index, article in enumerate(dataset, start=1):
        title = article["title"].strip()
        doc_type = determine_document_type(title)
        if doc_type is None:
            continue
        document_name = title
        if document_name in used_names:
            document_name = f"{title} [{article_index}]"
        used_names.add(document_name)
        paragraphs = article["paragraphs"]
        context = paragraphs[0]["context"]
        clauses = []
        for paragraph in paragraphs:
            if paragraph["context"] != context:
                raise ValueError(f"CUAD article has inconsistent contexts: {title!r}")
            for qa in paragraph["qas"]:
                clause_type = map_cuad_question_to_clause_type(qa["question"])
                if not clause_type:
                    continue
                for answer in qa["answers"]:
                    text = answer["text"]
                    start = answer["answer_start"]
                    end = start + len(text)
                    if not 0 <= start <= end <= len(context) or context[start:end] != text:
                        continue
                    clauses.append(
                        {
                            "clause_text": text,
                            "clause_type": clause_type,
                            "span_start": start,
                            "span_end": end,
                        }
                    )
        final_documents.append(
            {
                "document_name": document_name,
                "document_type": doc_type,
                "text": context,
                "clauses": clauses,
            }
        )

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
    """Print all distinct raw CUAD questions and their 41 category names."""
    del split, limit
    questions = sorted({
        qa["question"]
        for article in _load_cuad_articles()
        for paragraph in article["paragraphs"]
        for qa in paragraph["qas"]
    })
    print(f"Distinct raw CUAD questions ({len(questions)}):")
    for question in questions:
        print(f"  - {question}")
    categories = sorted({
        re.search(r'related to "([^"]+)"', question, flags=re.IGNORECASE).group(1)
        for question in questions
    })
    print(f"Full CUAD category list ({len(categories)}):")
    for category in categories:
        print(f"  - {category}")


def print_cuad_data_quality() -> None:
    """Print answer-span quality and document-title coverage diagnostics."""
    articles = _load_cuad_articles()
    checked = 0
    mismatches = 0
    for article in articles:
        context = article["paragraphs"][0]["context"]
        for qa in article["paragraphs"][0]["qas"]:
            if not map_cuad_question_to_clause_type(qa["question"]):
                continue
            for answer in qa["answers"]:
                if checked == 200:
                    break
                start = answer["answer_start"]
                text = answer["text"]
                end = start + len(text)
                checked += 1
                if not 0 <= start <= end <= len(context) or context[start:end] != text:
                    mismatches += 1
            if checked == 200:
                break
        if checked == 200:
            break
    mismatch_rate = mismatches / checked if checked else 0.0
    print(
        f"Answer-span validation sample: {checked} clauses, "
        f"{mismatches} mismatches ({mismatch_rate:.2%})"
    )

    titles_by_type: dict[str, list[str]] = collections.defaultdict(list)
    unmatched_titles = []
    for article in articles:
        title = article["title"].strip()
        doc_type = determine_document_type(title)
        if doc_type:
            titles_by_type[doc_type].append(title)
        else:
            unmatched_titles.append(title)
    rng = random.Random(0)
    print(
        f"CUAD contracts by title classification: matched "
        f"{len(articles) - len(unmatched_titles)}/{len(articles)}, "
        f"unmatched {len(unmatched_titles)}"
    )
    print(f"Random titles classified as None ({len(unmatched_titles)} total, up to 30):")
    for title in rng.sample(unmatched_titles, min(30, len(unmatched_titles))):
        print(f"  - {title}")
    title_counts = _cuad_title_classification_counts(articles)
    print(
        "Titles matched only by generic agreement/contract keywords:",
        title_counts["generic_only"],
    )
    print(
        "True title matches for NDA, employment, rental:",
        {
            doc_type: title_counts[doc_type]
            for doc_type in ("nda", "employment", "rental")
        },
    )
    print("Contracts excluded because title matched nothing:", title_counts["unmatched"])
    for doc_type in DOCUMENT_TYPES:
        titles = titles_by_type.get(doc_type, [])
        print(f"20 random titles classified as {doc_type} ({len(titles)} total):")
        for title in rng.sample(titles, min(20, len(titles))):
            print(f"  - {title}")


def write_dataset_report(documents: list[dict[str, Any]], path: str = "DATASET_REPORT.md") -> None:
    """Write final coverage counts and the requested data-quality summary."""
    document_counts = collections.Counter(document["document_type"] for document in documents)
    clause_counts = collections.Counter(
        clause["clause_type"]
        for document in documents
        for clause in document.get("clauses", [])
    )
    zero = [name for name in CLAUSE_TYPES if clause_counts[name] == 0]
    low = [name for name in CLAUSE_TYPES if 0 < clause_counts[name] < 20]
    with open(path, "w", encoding="utf-8") as report:
        report.write("# Dataset report\n\n")
        report.write("## Document counts per type\n\n")
        report.write("| Document type | Count |\n|---|---:|\n")
        for name in DOCUMENT_TYPES:
            report.write(f"| {name} | {document_counts[name]} |\n")
        report.write("\n## Clause counts per type\n\n")
        report.write("| Clause type | Count |\n|---|---:|\n")
        for name in CLAUSE_TYPES:
            report.write(f"| {name} | {clause_counts[name]} |\n")
        report.write("\n## Missing and low-coverage clause types\n\n")
        report.write(f"- Zero examples: {', '.join(zero) or 'None'}\n")
        report.write(f"- Fewer than 20 examples: {', '.join(low) or 'None'}\n")
        cuad_title_counts = _cuad_title_classification_counts(_load_cuad_articles())
        report.write("\n## Title-based document-type facts\n\n")
        report.write(
            "The title heuristic found "
            f"{cuad_title_counts['nda']} NDA, {cuad_title_counts['employment']} employment, "
            f"and {cuad_title_counts['rental']} rental contracts. "
            f"{cuad_title_counts['generic_only']} contracts were labeled only by generic "
            "agreement/contract keywords. "
            f"{cuad_title_counts['unmatched']} contracts were excluded because their "
            "titles matched nothing.\n"
        )
        report.write("\n## Changes and remaining problems\n\n")
        report.write(
            "- Loaded only CUAD contracts whose titles matched a document-type keyword and "
            "skipped contracts whose titles matched nothing.\n"
            "- Used word-boundary regexes with specific title signals before generic agreement "
            "and contract signals.\n"
            "- Kept only CUAD categories that directly represent our clause types; confidentiality, "
            "indemnification, payment_terms, data_processing, dispute_resolution, and "
            "force_majeure remain unmapped because CUAD has no matching category.\n"
            "- CUAD has very few title-identifiable NDA, employment, and rental contracts; "
            "generic agreement/contract matches are reported separately and are not treated "
            "as stronger evidence.\n"
            "- Remaining problem: title heuristics exclude contracts with no matching title "
            "keyword, so the filtered dataset is not all CUAD contracts.\n"
        )


if __name__ == "__main__":
    docs = load_and_filter_cuad_dataset()
    print(f"Loaded {len(docs)} documents matching our criteria.")
    print_dataset_diagnostics(docs)
    print_cuad_question_examples()
    print_cuad_data_quality()
    validate_documents(docs)
    print("validate_documents passed.")
    write_dataset_report(docs)
