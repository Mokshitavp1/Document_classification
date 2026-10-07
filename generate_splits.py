import json
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split
from dataset_prep import get_full_dataset

def main():
    print("Loading full dataset to extract document names...")
    try:
        dataset = get_full_dataset()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Ensure Windows Long Paths are enabled and you have generated bootstrap_labels_corrected.json.")
        return

    documents_by_type = defaultdict(list)
    for doc in dataset:
        documents_by_type[doc["document_type"]].append(doc["document_name"])
    doc_names = list(set(doc["document_name"] for doc in dataset))
    print(f"Found {len(doc_names)} unique documents.")
    
    if not doc_names:
        print("No documents found. Cannot generate splits.")
        return
        
    splits = {"train": [], "val": [], "test": []}
    for document_type, names in sorted(documents_by_type.items()):
        names = sorted(names)
        if len(names) < 3:
            print(f"Warning: {document_type} has fewer than 3 documents; assigning all to train.")
            splits["train"].extend(names)
            continue
        train_names, remainder = train_test_split(
            names, test_size=0.30, random_state=42
        )
        val_names, test_names = train_test_split(
            remainder, test_size=0.50, random_state=42
        )
        splits["train"].extend(train_names)
        splits["val"].extend(val_names)
        splits["test"].extend(test_names)
    for values in splits.values():
        values.sort()
    
    output = Path("splits.json")
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != splits:
            raise RuntimeError(
                "splits.json already exists and differs. Delete it explicitly only "
                "if you intend to establish a new reproducibility contract."
            )
    else:
        output.write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")
        
    print(f"Generated splits.json successfully!")
    print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

if __name__ == "__main__":
    main()
