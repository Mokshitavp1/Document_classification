"""Train document classification and token-level clause extraction models."""

from __future__ import annotations

import argparse
import inspect
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score
from seqeval.metrics import f1_score as entity_f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from contract_constants import CLAUSE_TYPES, DOCUMENT_TYPES
from dataset_prep import get_full_dataset
from windowing import assign_bio_labels

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
                max_length=512, stride=128, return_overflowing_tokens=True,
                return_offsets_mapping=True, return_special_tokens_mask=True,
                return_tensors="pt",
            )
            window_count = encoded["input_ids"].shape[0]
            for window_index in range(window_count):
                offsets = encoded["offset_mapping"][window_index].tolist()
                masks = encoded["special_tokens_mask"][window_index].tolist()
                attention = encoded["attention_mask"][window_index].tolist()
                labels = assign_bio_labels(
                    offsets, document["clauses"], label2id,
                    special_tokens_mask=masks, attention_mask=attention,
                )
                item = {
                    key: value[window_index]
                    for key, value in encoded.items()
                    if key not in {"offset_mapping", "special_tokens_mask", "overflow_to_sample_mapping"}
                }
                item["labels"] = torch.tensor(labels)
                self.items.append(item)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.items[index]


def _evaluation_strategy_kwargs() -> dict[str, str]:
    parameter = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in parameter:
        return {"eval_strategy": "epoch"}
    return {"evaluation_strategy": "epoch"}


def _label_strings(predictions: Any, labels: Any, id2label: dict[int, str]) -> tuple[list[list[str]], list[list[str]]]:
    predicted_ids = np.argmax(predictions, axis=-1)
    predicted_strings: list[list[str]] = []
    gold_strings: list[list[str]] = []
    for predicted_row, gold_row in zip(predicted_ids, labels):
        pred_sequence: list[str] = []
        gold_sequence: list[str] = []
        for predicted_id, gold_id in zip(predicted_row, gold_row):
            if gold_id == -100:
                continue
            pred_sequence.append(id2label[int(predicted_id)])
            gold_sequence.append(id2label[int(gold_id)])
        predicted_strings.append(pred_sequence)
        gold_strings.append(gold_sequence)
    return predicted_strings, gold_strings


def document_compute_metrics(eval_prediction: Any) -> dict[str, float]:
    predictions, labels = eval_prediction
    predicted_ids = np.argmax(predictions, axis=-1)
    return {"macro_f1": float(f1_score(labels, predicted_ids, average="macro"))}


def clause_compute_metrics(eval_prediction: Any, id2label: dict[int, str]) -> dict[str, float]:
    predictions, labels = eval_prediction
    predicted, gold = _label_strings(predictions, labels, id2label)
    return {"entity_f1": float(entity_f1_score(gold, predicted))}


def clause_class_weights(dataset: ClauseDataset, label2id: dict[str, int]) -> torch.Tensor:
    counts = torch.zeros(len(label2id), dtype=torch.float)
    for item in dataset:
        for label in item["labels"].tolist():
            if label != -100:
                counts[label] += 1
    weights = torch.ones_like(counts)
    nonzero = counts > 0
    weights[nonzero] = counts[nonzero].sum() / (nonzero.sum() * counts[nonzero])
    weights.clamp_(max=10.0)
    weights[label2id["O"]] *= 0.25
    return weights


class WeightedTokenTrainer(Trainer):
    def __init__(self, *args: Any, class_weights: torch.Tensor, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False, **kwargs: Any) -> Any:
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(outputs.logits.device), ignore_index=-100
        )(outputs.logits.view(-1, outputs.logits.shape[-1]), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=Path, default=Path("splits.json"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/legal_bert_clause_v1"))
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--max-train-docs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    set_seed()
    documents = get_full_dataset()
    split_documents = build_training_examples(documents, load_splits(args.splits))
    if args.max_train_docs is not None:
        split_documents["train"] = split_documents["train"][:args.max_train_docs]
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
        num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size, **_evaluation_strategy_kwargs(),
        save_strategy="no", report_to="none", seed=SEED,
    )
    document_trainer = Trainer(
        model=document_model, args=document_args,
        train_dataset=DocumentDataset(split_documents["train"], tokenizer),
        eval_dataset=DocumentDataset(split_documents["val"], tokenizer),
        compute_metrics=document_compute_metrics,
    )
    document_trainer.train()
    document_model.save_pretrained(args.output / "document_classifier")

    clause_args = TrainingArguments(
        output_dir=str(args.output / "clause_training"),
        num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size, **_evaluation_strategy_kwargs(),
        save_strategy="no", report_to="none", seed=SEED,
    )
    clause_dataset = ClauseDataset(split_documents["train"], tokenizer, clause_model.config.label2id)
    clause_trainer = WeightedTokenTrainer(
        model=clause_model, args=clause_args,
        train_dataset=clause_dataset,
        eval_dataset=ClauseDataset(split_documents["val"], tokenizer, clause_model.config.label2id),
        compute_metrics=lambda prediction: clause_compute_metrics(
            prediction, clause_model.config.id2label
        ),
        class_weights=clause_class_weights(clause_dataset, clause_model.config.label2id),
    )
    clause_trainer.train()
    clause_model.save_pretrained(args.output / "clause_extractor")
    tokenizer.save_pretrained(args.output)
    (args.output / "training_manifest.json").write_text(
        json.dumps({
            "base_model": BASE_MODEL, "revision": REVISION, "seed": SEED,
            "epochs": args.epochs,
            "transformers_version": __import__("transformers").__version__,
            "torch_version": torch.__version__,
            "counts": {key: len(value) for key, value in split_documents.items()},
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Saved fine-tuned checkpoint to {args.output}")


if __name__ == "__main__":
    main()
