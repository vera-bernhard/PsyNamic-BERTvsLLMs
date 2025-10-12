import argparse
import json
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_model, LoraConfig, TaskType
from billm import LlamaForTokenClassification
from nervaluate import Evaluator
import numpy as np
import wandb
import os


# based on: https://github.com/WhereIsAI/BiLLM/blob/main/examples/billm_ner.py


parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str,
                    default="meta-llama/Llama-3.1-8B-Instruct")
parser.add_argument("--data_dir", type=str, default="./data/ner_bio")
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--max_length", type=int, default=800)
parser.add_argument("--lora_r", type=int, default=32)
parser.add_argument("--lora_alpha", type=int, default=32)
parser.add_argument("--lora_dropout", type=float, default=0.1)
parser.add_argument("--output_dir", type=str,
                    default="./label_supervised_model")

args = parser.parse_args()


print("Loading dataset and meta.json...")
with open(f"{args.data_dir}/meta.json", "r") as f:
    meta = json.load(f)

int_to_label = {int(k): v for k, v in meta["Int_to_label"].items()}
label2id = {v: k for k, v in int_to_label.items()}
id2label = {k: v for v, k in label2id.items()}

# wandb init
wandb.init(project="PsyNamic-LS-Tune", name=args.output_dir.split("/")[-1])

def load_split(split_name):
    df = pd.read_csv(f"{args.data_dir}/{split_name}.csv")
    df["tokens"] = df["tokens"].apply(eval)
    df["ner_tags"] = df["ner_tags"].apply(eval)
    return Dataset.from_pandas(df)


ds = DatasetDict({
    "train": load_split("train"),
    "validation": load_split("val"),
    "test": load_split("test"),
})

print(ds)
# print some everage, max and min lengths of tokens in the dataset
all_lengths = [len(x) for x in ds["train"]["tokens"]]
print(f"Average length: {np.mean(all_lengths)}")
print(f"Max length: {np.max(all_lengths)}")
print(f"Min length: {np.min(all_lengths)}")


print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

model = LlamaForTokenClassification.from_pretrained(
    args.model_name_or_path,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
).bfloat16()

peft_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS,
    inference_mode=False,
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=args.max_length,
        padding="max_length",
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            else:
                # Label-supervised (no masking): assign label directly to all subtokens
                label_ids.append(label2id[label[word_idx]])
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


tokenized_ds = ds.map(tokenize_and_align_labels, batched=True)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


evaluator = Evaluator(tag_scheme="BIO")


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2label[p] for (p, l) in zip(pred_row, label_row) if l != -100]
        for pred_row, label_row in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(pred_row, label_row) if l != -100]
        for pred_row, label_row in zip(predictions, labels)
    ]

    results, _ = evaluator.evaluate(true_predictions, true_labels)

    # Return overall metrics
    return {
        "precision": results["precision"],
        "recall": results["recall"],
        "f1": results["f1"],
    }

training_args = TrainingArguments(
    output_dir=args.output_dir,
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    weight_decay=args.weight_decay,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir=f"{args.output_dir}/logs",
    logging_steps=50,
    load_best_model_at_end=True,
    report_to="wandb",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Starting fine-tuning...")
trainer.train()

wandb.finish()

os.makedirs(args.output_dir, exist_ok=True)

# save best model
print("Saving best model...")
trainer.save_model(args.output_dir)

print("Evaluating best model and predicting on test set...")
pred_output = trainer.predict(tokenized_ds["test"])
metrics = pred_output.metrics
# save metrics to a json file
with open(f"{args.output_dir}/test_metrics.json", "w") as f:
    json.dump(metrics, f)

predictions = np.argmax(pred_output.predictions, axis=2)
test_labels = pred_output.label_ids

test_tokens = ds["test"]["tokens"]
test_ids = ds["test"]["id"] if "id" in ds["test"].column_names else list(range(len(test_tokens)))
pred_labels = [
    [id2label[p] for (p, l) in zip(pred_row, label_row) if l != -100]
    for pred_row, label_row in zip(predictions, test_labels)
]

df_pred = pd.DataFrame({
    "id": test_ids,
    "tokens": [str(toks) for toks in test_tokens],
    "pred_labels": [str(labels) for labels in pred_labels],
})
df_pred.to_csv(f"{args.output_dir}/test_predictions.csv", index=False)
print(f"Predictions saved to {args.output_dir}/test_predictions.csv")
