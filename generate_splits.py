import json
import random
from dataset_prep import get_full_dataset

def main():
    print("Loading full dataset to extract document names...")
    try:
        dataset = get_full_dataset()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Ensure Windows Long Paths are enabled and you have generated bootstrap_labels_corrected.json.")
        return

    doc_names = list(set(doc["document_name"] for doc in dataset))
    print(f"Found {len(doc_names)} unique documents.")
    
    if not doc_names:
        print("No documents found. Cannot generate splits.")
        return
        
    doc_names.sort()
    random.seed(42)
    random.shuffle(doc_names)
    
    n = len(doc_names)
    train_end = int(n * 0.70)
    val_end = train_end + int(n * 0.15)
    
    splits = {
        "train": doc_names[:train_end],
        "val": doc_names[train_end:val_end],
        "test": doc_names[val_end:]
    }
    
    with open("splits.json", "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
        
    print(f"Generated splits.json successfully!")
    print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

if __name__ == "__main__":
    main()
