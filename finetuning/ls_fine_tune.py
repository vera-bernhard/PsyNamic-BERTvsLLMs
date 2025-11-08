import argparse
import json
import pandas as pd
from ast import literal_eval
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    set_seed
)
from peft import get_peft_model, LoraConfig, TaskType
from billm import LlamaForTokenClassification
from nervaluate import Evaluator
import numpy as np
import wandb
import os
from huggingface_hub import login
from dotenv import load_dotenv
import torch
from nervaluate import Evaluator



load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
access_token = os.getenv("ACCESS_TOKEN")
login(access_token)
wandb.login(key=os.getenv("WANDB"))


SEED = 42
# set seeds
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
np.random.seed(SEED)
set_seed(SEED)

# based on: https://github.com/WhereIsAI/BiLLM/blob/main/examples/billm_ner.py


parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str,
                    default="meta-llama/Llama-3.1-8B-Instruct")
parser.add_argument("--data_dir", type=str, default="./data/ner_bio")
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--max_length", type=int, default=1200)
parser.add_argument("--lora_r", type=int, default=32)
parser.add_argument("--lora_alpha", type=int, default=32)
parser.add_argument("--lora_dropout", type=float, default=0.1)
parser.add_argument("--output_dir", type=str,
                    default=os.path.join(os.path.dirname(__file__), "lst_llama3_8b_2.0"))

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
    df = df.drop(columns=["bert_tokens", "word_ids", "bert_ner_tags"], errors='ignore')
    return Dataset.from_pandas(df)


ds = DatasetDict({
    "train": load_split("train"),
    "validation": load_split("val"),
    "test": load_split("test"),
})

# print some everage, max and min lengths of tokens in the dataset
all_lengths = [len(x) for x in ds["train"]["tokens"]]
print(f"Average length: {np.mean(all_lengths)}")
print(f"Max length: {np.max(all_lengths)}")
print(f"Min length: {np.min(all_lengths)}")


print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = LlamaForTokenClassification.from_pretrained(
    args.model_name_or_path,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
    # device_map="auto",
    low_cpu_mem_usage=True,
)

peft_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS,
    inference_mode=False,
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

print("num_labels in model:", model.config.num_labels)


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        padding="longest",
        max_length=args.max_length,
        truncation=True,
    )

    labels = []
    word_ids_all = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        word_ids_all.append(word_ids)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    tokenized_inputs["word_ids"] = word_ids_all
    return tokenized_inputs


tokenized_ds = ds.map(tokenize_and_align_labels, batched=True)
# some stats on whole long the dataset tokenized labels
print("Tokenized dataset label lengths stats:")
all_label_lengths = []
for i in range(len(tokenized_ds["train"])):
    all_label_lengths.append(len(tokenized_ds["train"][i]["labels"]))
print(f"Average length: {np.mean(all_label_lengths)}")
print(f"Max length: {np.max(all_label_lengths)}")
print(f"Min length: {np.min(all_label_lengths)}")

# for test too
all_label_lengths_test = []
for i in range(len(tokenized_ds["test"])):
    all_label_lengths_test.append(len(tokenized_ds["test"][i]["labels"]))
print(f"Average length: {np.mean(all_label_lengths_test)}")
print(f"Max length: {np.max(all_label_lengths_test)}")
print(f"Min length: {np.min(all_label_lengths_test)}")

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Look at some tokenized labels to check for invalid values
print("Sample tokenized labels from train set:")
for i in range(min(3, len(tokenized_ds["train"]))):
    print(tokenized_ds["train"][i]["labels"])


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

    # Based on nervaluate 0.2.0 because that is only supported with Python 3.9
    evaluator = Evaluator(true=true_labels, pred=true_predictions, tags=['Application area', 'Dosage'], loader='list')
    results, resultsagg, _, _ = evaluator.evaluate()
    r = {
        'f1 overall - strict': results['strict']['f1'],
        'f1 overall - partial': results['partial']['f1'],
    }

    return r


training_args = TrainingArguments(
    output_dir=args.output_dir,
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    weight_decay=args.weight_decay,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    # save_strategy="steps",
    # save_steps=20,
    logging_dir=f"{args.output_dir}/logs",
    logging_steps=50,
    load_best_model_at_end=True,
    report_to="wandb",
    seed=SEED,
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
os.makedirs(args.output_dir, exist_ok=True)

# save best model
print("Saving best model...")
print("Best model checkpoint:", trainer.state.best_model_checkpoint)

trainer.save_model(args.output_dir)

print("Evaluating best model and predicting on test set...")
pred_output = trainer.predict(tokenized_ds["test"])
pred_logits = pred_output.predictions
pred_ids = np.argmax(pred_logits, axis=-1)

pred_labels = []
for i in range(len(tokenized_ds["test"])):
    word_ids = tokenized_ds["test"][i]['word_ids']
    preds = []
    prev_wid = None
    for pid, wid in zip(pred_ids[i], word_ids):
        if wid is None:
            continue
        if wid != prev_wid:
            preds.append(id2label[int(pid)])
        prev_wid = wid
    pred_labels.append(preds)

# original tokens
test_tokens = ds["test"]["tokens"]
test_ids = ds["test"]["id"]
text = ds["test"]["text"]
ner_tags = ds["test"]["ner_tags"]

# Check if ner_tags length matches pred_labels length
for i in range(len(ner_tags)):
    if len(ner_tags[i]) != len(pred_labels[i]):
        print(f"Example {i} has mismatched lengths: gold({len(ner_tags[i])}) vs pred({len(pred_labels[i])})")

df_pred = pd.DataFrame({
    "id": test_ids,
    "tokens": test_tokens,
    "pred_labels": pred_labels,
    "text": text,
    "ner_tags": ner_tags
})
df_pred.to_csv(f"{args.output_dir}/test_predictions.csv", index=False)

wandb.finish()
