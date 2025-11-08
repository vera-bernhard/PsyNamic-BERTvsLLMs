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
from peft import PeftModel, PeftConfig
from evaluation.evaluate import evaluate_ner_bio  # not used but keeps parity
# disable wandb integration during prediction
os.environ["WANDB_MODE"] = "offline"

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="./finetuning/lst_llama3_8b")
parser.add_argument("--data_dir", type=str, default="./data/ner_bio")
parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
parser.add_argument("--batch_size", type=int, default=4)
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
df_test = pd.read_csv(os.path.join(args.data_dir, "val.csv"))
df_test["tokens"] = df_test["tokens"].apply(eval)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# load model: try full model first, then try PEFT adapter on top of base
# 1) try loading full model from model_dir
peft_config = PeftConfig.from_pretrained(args.model_dir)

base = LlamaForTokenClassification.from_pretrained(
    peft_config.base_model_name_or_path,
    num_labels=len(label2id),
    id2label={k: v for k, v in id2label.items()},
    label2id=label2id,
    low_cpu_mem_usage=False,
)
test_text = "This is a test input"
inputs = tokenizer(test_text, return_tensors="pt") 
model = PeftModel.from_pretrained(base, args.model_dir)
model = model.merge_and_unload()
print(model.peft_config)

model.to(device)
model.eval()

for i, row in df_test.iterrows():
    inputs = tokenizer(
        row["tokens"],
        is_split_into_words=True,
        return_tensors="pt",
        truncation=False,
    )
    bert_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    word_ids = inputs.word_ids(batch_index=0)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_ids = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()  # shape (seq_len,)

    preds = []
    prev_wid = None
    for pid, wid in zip(pred_ids, word_ids):
        if wid is None:
            continue
        if wid != prev_wid:
            preds.append(id2label[int(pid)])
        prev_wid = wid

    gold_len = len(row["tokens"])
    if gold_len != len(preds):
       print(f"Row {i} has mismatched lengths: gold({gold_len}) vs pred({len(preds)})")

    df_test.at[i, "pred_labels"] = str(preds)
    df_test.at[i, "bert_tokens"] = str(bert_tokens)
    df_test.at[i, "bert_ner_tags"] = str([id2label[int(pid)] for pid in pred_ids])
    df_test.at[i, "word_ids"] = str(word_ids)

    
out_path = os.path.join(args.model_dir, "val_predictions.csv")
df_test.to_csv(out_path, index=False)
