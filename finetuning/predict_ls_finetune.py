import os
import json
import argparse
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)
from billm import LlamaForTokenClassification
from peft import PeftModel
from evaluation.evaluate import evaluate_ner_bio  # not used but keeps parity
# disable wandb integration during prediction
os.environ["WANDB_MODE"] = "offline"

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="./finetuning/lst_llama3_8b")
parser.add_argument("--data_dir", type=str, default="./data/ner_bio")
parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--max_length", type=int, default=800)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

set_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load label mapping
with open(os.path.join(args.data_dir, "meta.json"), "r") as f:
    meta = json.load(f)
int_to_label = {int(k): v for k, v in meta["Int_to_label"].items()}
label2id = {v: k for k, v in int_to_label.items()}
id2label = {k: v for k, v in int_to_label.items()}

# load test csv (same loader as in ls_fine_tune.py)
df_test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
df_test["tokens"] = df_test["tokens"].apply(eval)
# create minimal Dataset
test_ds = Dataset.from_pandas(df_test)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# load model: try full model first, then try PEFT adapter on top of base
model = None
# 1) try loading full model from model_dir
try:
    model = LlamaForTokenClassification.from_pretrained(args.model_dir)
    print("Loaded full model from", args.model_dir)
except Exception:
    # 2) try loading base and then the PEFT adapter from model_dir
    try:
        base = LlamaForTokenClassification.from_pretrained(
            args.model_name_or_path,
            num_labels=len(label2id),
            id2label={k: v for k, v in id2label.items()},
            label2id=label2id,
            low_cpu_mem_usage=False,
        )
        model = PeftModel.from_pretrained(base, args.model_dir, is_trainable=False)
        print("Loaded base model and PEFT adapter from", args.model_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {args.model_dir}: {e}")

model.to(device)
model.eval()

# tokenize test dataset for Trainer (batched)
def tokenize_for_trainer(batch):
    return tokenizer(
        batch["tokens"],
        is_split_into_words=True,
        padding="longest",
        truncation=True,
        max_length=args.max_length,
    )

tokenized_test = test_ds.map(tokenize_for_trainer, batched=True)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# create Trainer with no integrations
predict_args = TrainingArguments(
    output_dir="tmp_predict",
    per_device_eval_batch_size=args.batch_size,
    report_to=None,
)

trainer = Trainer(
    model=model,
    args=predict_args,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("Running prediction on test set...")
pred_output = trainer.predict(tokenized_test)
log_metrics = pred_output.metrics
print("Predict metrics:", log_metrics)

pred_logits = pred_output.predictions  # shape (n_examples, seq_len, num_labels)
pred_ids = np.argmax(pred_logits, axis=-1)

# map predictions back to word-level labels using tokenizer.word_ids per example
pred_labels = []
for i in range(len(tokenized_test)):
    word_ids = tokenized_test[i].word_ids()
    preds = []
    prev_wid = None
    for pid, wid in zip(pred_ids[i], word_ids):
        if wid is None:
            continue
        if wid != prev_wid:
            preds.append(id2label[int(pid)])
        prev_wid = wid
    pred_labels.append(preds)

    gold_len = len(df_test["tokens"][i])
    if gold_len != len(preds):
       print(f"Row {i} has mismatched lengths: gold({gold_len}) vs pred({len(preds)})")

out_df = pd.DataFrame({
    "id": df_test["id"].tolist(),
    "tokens": df_test["tokens"].apply(str).tolist(),
    "pred_labels": [str(p) for p in pred_labels],
    "ner_tags": df_test["ner_tags"].tolist(),
})


os.makedirs(args.model_dir, exist_ok=True)
out_path = os.path.join(args.model_dir, "test_predictions.csv")
out_df.to_csv(out_path, index=False)
print("Saved predictions to", out_path)