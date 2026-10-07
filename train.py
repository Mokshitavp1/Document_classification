"""Train document classification and token-level clause extraction models."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from contract_constants import CLAUSE_TYPES, DOCUMENT_TYPES
from dataset_prep import get_full_dataset

SEED = 42
BASE_MODEL = "nlpaueb/legal-bert-base-uncased"
REVISION = "15b570cbf88259610b082a167dacc190124f60f6"


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def load_splits(path: Path) -> dict[str, list[str]]:
    splits = json.loads(path.read_text(encoding="utf-8"))
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("splits.json must contain train, val, and test")
    ids = [item for values in splits.values() for item in values]
    if len(ids) != len(set(ids)):
        raise ValueError("splits.json contains duplicate document IDs")
    return splits


def build_training_examples(documents: list[dict], splits: dict[str, list[str]]) -> dict[str, list[dict]]:
    by_name = {doc["document_name"]: doc for doc in documents}
    missing = set(sum(splits.values(), [])) - set(by_name)
    if missing:
        raise ValueError(f"splits.json references missing documents, e.g. {next(iter(missing))}")
    return {split: [by_name[name] for name in names] for split, names in splits.items()}


class DocumentDataset(torch.utils.data.Dataset):
    def __init__(self, documents: list[dict], tokenizer: AutoTokenizer):
        self.items = []
        for document in documents:
            encoded = tokenizer(
                document["text"], truncation=True, padding="max_length",
                max_length=512, return_tensors="pt",
            )
            item = {key: value.squeeze(0) for key, value in encoded.items()}
            item["labels"] = torch.tensor(DOCUMENT_TYPES.index(document["document_type"]))
            self.items.append(item)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.items[index]


class ClauseDataset(torch.utils.data.Dataset):
    def __init__(self, documents: list[dict], tokenizer: AutoTokenizer, label2id: dict[str, int]):
        self.items = []
        for document in documents:
            encoded = tokenizer(
                document["text"], truncation=True, padding="max_length",
                max_length=512, return_offsets_mapping=True, return_tensors="pt",
            )
            offsets = encoded.pop("offset_mapping")[0].tolist()
            labels = []
            for start, end in offsets:
                label = "O"
                for clause in document["clauses"]:
                    if start < clause["span_end"] and end > clause["span_start"]:
                        prefix = "B" if start <= clause["span_start"] else "I"
                        label = f"{prefix}-{clause['clause_type']}"
                        break
                labels.append(label2id[label] if start != end else -100)
            item = {key: value.squeeze(0) for key, value in encoded.items()}
            item["labels"] = torch.tensor(labels)
            self.items.append(item)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.items[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=Path, default=Path("splits.json"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/legal_bert_clause_v1"))
    parser.add_argument("--epochs", type=float, default=3)
    args = parser.parse_args()
    set_seed()
    documents = get_full_dataset()
    split_documents = build_training_examples(documents, load_splits(args.splits))
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=REVISION)
    document_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, revision=REVISION, num_labels=len(DOCUMENT_TYPES),
        id2label={i: label for i, label in enumerate(DOCUMENT_TYPES)},
        label2id={label: i for i, label in enumerate(DOCUMENT_TYPES)},
    )
    clause_labels = ["O"] + [f"B-{label}" for label in CLAUSE_TYPES] + [f"I-{label}" for label in CLAUSE_TYPES]
    clause_model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL, revision=REVISION, num_labels=len(clause_labels),
        id2label=dict(enumerate(clause_labels)),
        label2id={label: i for i, label in enumerate(clause_labels)},
    )
    args.output.mkdir(parents=True, exist_ok=True)
    document_args = TrainingArguments(
        output_dir=str(args.output / "document_training"),
        num_train_epochs=args.epochs, per_device_train_batch_size=2,
        per_device_eval_batch_size=2, evaluation_strategy="epoch",
        save_strategy="no", report_to="none", seed=SEED,
    )
    document_trainer = Trainer(
        model=document_model, args=document_args,
        train_dataset=DocumentDataset(split_documents["train"], tokenizer),
        eval_dataset=DocumentDataset(split_documents["val"], tokenizer),
    )
    document_trainer.train()
    document_model.save_pretrained(args.output / "document_classifier")

    clause_args = TrainingArguments(
        output_dir=str(args.output / "clause_training"),
        num_train_epochs=args.epochs, per_device_train_batch_size=2,
        per_device_eval_batch_size=2, evaluation_strategy="epoch",
        save_strategy="no", report_to="none", seed=SEED,
    )
    clause_trainer = Trainer(
        model=clause_model, args=clause_args,
        train_dataset=ClauseDataset(split_documents["train"], tokenizer, clause_model.config.label2id),
        eval_dataset=ClauseDataset(split_documents["val"], tokenizer, clause_model.config.label2id),
    )
    clause_trainer.train()
    clause_model.save_pretrained(args.output / "clause_extractor")
    tokenizer.save_pretrained(args.output)
    (args.output / "training_manifest.json").write_text(
        json.dumps({"base_model": BASE_MODEL, "revision": REVISION, "seed": SEED,
                    "counts": {key: len(value) for key, value in split_documents.items()}}, indent=2),
        encoding="utf-8",
    )
    print("Checkpoint components initialized. Full Trainer fine-tuning requires tokenized datasets.")


if __name__ == "__main__":
    main()
