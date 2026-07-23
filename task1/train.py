import argparse
import json
import os

import numpy as np
from datasets import Dataset, DatasetDict
from seqeval.metrics import classification_report, f1_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

LABELS = ["O", "B-MOUNTAIN", "I-MOUNTAIN"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def make_hf_dataset(examples: list[dict]) -> Dataset:
    return Dataset.from_dict({
        "tokens": [ex["tokens"] for ex in examples],
        "ner_tags": [[LABEL2ID[t] for t in ex["tags"]] for ex in examples],
    })


def tokenize_and_align_labels(examples: dict, tokenizer) -> dict:
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
    )
    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev_word_id = None
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                label_ids.append(labels[word_id])
            else:
                label_ids.append(-100)
            prev_word_id = word_id
        all_labels.append(label_ids)
    tokenized["labels"] = all_labels
    return tokenized


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    true_seqs, pred_seqs = [], []
    for pred_row, label_row in zip(predictions, labels):
        true_seq, pred_seq = [], []
        for p, l in zip(pred_row, label_row):
            if l != -100:
                true_seq.append(ID2LABEL[l])
                pred_seq.append(ID2LABEL[p])
        true_seqs.append(true_seq)
        pred_seqs.append(pred_seq)
    return {
        "f1": f1_score(true_seqs, pred_seqs),
        "report": classification_report(true_seqs, pred_seqs),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="bert-base-cased")
    parser.add_argument("--data_dir", default="dataset")
    parser.add_argument("--output_dir", default="model_output")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_examples = load_jsonl(os.path.join(args.data_dir, "train.jsonl"))
    val_examples   = load_jsonl(os.path.join(args.data_dir, "val.jsonl"))
    test_examples  = load_jsonl(os.path.join(args.data_dir, "test.jsonl"))

    raw_datasets = DatasetDict({
        "train":      make_hf_dataset(train_examples),
        "validation": make_hf_dataset(val_examples),
        "test":       make_hf_dataset(test_examples),
    })
    print(f"train: {len(raw_datasets['train'])}  val: {len(raw_datasets['validation'])}  test: {len(raw_datasets['test'])}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenized_datasets = raw_datasets.map(
        lambda ex: tokenize_and_align_labels(ex, tokenizer),
        batched=True,
        remove_columns=["tokens", "ner_tags"],
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=10,
        seed=args.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    test_metrics = trainer.evaluate(tokenized_datasets["test"])
    print("\n=== Test set ===")
    print(test_metrics.get("eval_report", ""))
    print(f"F1: {test_metrics.get('eval_f1', 0):.4f}")

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nSaved to '{args.output_dir}/'")


if __name__ == "__main__":
    main()
