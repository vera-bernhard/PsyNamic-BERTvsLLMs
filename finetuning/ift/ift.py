"""
Filename: ift_predict.py
Description: Script to fine-tune language models using instruction fine-tuning with PEFT and SFTTrainer.
Author: Vera Bernhard
"""


# based on:
# https://huggingface.co/docs/trl/sft_trainer
# https://www.datacamp.com/tutorial/llama3-fine-tuning-locally?dc_referrer=https%3A%2F%2Fwww.google.com%2F

import os
import json
import ast
import random
from typing import Optional

import torch
import pandas as pd
from datasets import load_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, prepare_model_for_kbit_training

import wandb

from prompts.build_prompts import (
    build_class_prompt,
    system_role_class,
    system_role_ner,
    build_ner_prompt,
    build_json_labels,
    get_int2label,
    get_class_options,
    markup_entities
)

from zero_shot.icl_predict import TASKS

SEED = 42
# set seeds
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

# Prompt completion dataset
#  {"prompt": "The sky is",
#  "completion": " blue."}


def build_instruction_finetune_dataset(
    data_dir: str,
    tasks: list[str],
    tokenizer,
    split: str = "train",
) -> list[dict[str, str]]:
    """
    Aggregate all splits of all classification tasks into a single shuffled
    instruction tuning dataset.
    Each example is a dict with a single 'text' field containing the flattened chat conversation.
    """
    random.seed(SEED)
    dataset = []

    for task in tasks:
        task_lower = task.lower().replace(" ", "_")
        split_path = os.path.join(data_dir, task_lower, f"{split}.csv")
        if not os.path.exists(split_path):
            print(f"Warning: {split_path} does not exist, skipping.")
            continue

        df = pd.read_csv(split_path)

        if "ner" in task_lower:
            for _, row in df.iterrows():
                prompt = build_ner_prompt(
                    row["id"],
                    row["text"],
                    few_shot=0,
                    few_shot_strategy="selected",
                )
                system_prompt = system_role_ner

                tokens = ast.literal_eval(row["tokens"])
                labels = ast.literal_eval(row["ner_tags"])
                text = row["text"]
                output = markup_entities(tokens, text, labels)

                # Construct chat
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": output},
                ]
                flattened = tokenizer.apply_chat_template(
                    messages, tokenize=False)
                dataset.append({"text": flattened})

        else:
            int2label = get_int2label(task)
            for _, row in df.iterrows():
                prompt = build_class_prompt(
                    row["id"],
                    task,
                    row["text"],
                    few_shot=0,
                    few_shot_strategy="selected",
                )
                if "ner" in task_lower:
                    system_prompt = system_role_ner
                else:
                    system_prompt = system_role_class

                options = get_class_options(task)
                output_json = build_json_labels(
                    row["labels"], int2label, options)

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": output_json},
                ]
                flattened = tokenizer.apply_chat_template(
                    messages, tokenize=False)
                dataset.append({"text": flattened})

    random.shuffle(dataset)
    return dataset


def sft_peft(model_name: str, data_file: str, output_dir: str, resume_from_checkpoint: Optional[str] = None):
    """
    Fine-tune a language model using SFTTrainer and PEFT QLoRA.
    """

    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files={"train": data_file})
    train_dataset = dataset["train"]

    bnb_kwargs = {}

    # Use 4-bit quantization for large models
    bnb_kwargs.update(
        dict(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16 if not '70b' in model_name else None,
        low_cpu_mem_usage=True,
        **bnb_kwargs,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    wandb.init(project='PsyNamic-Scale-SFT', name=os.path.basename(output_dir))

    training_args = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        num_train_epochs=3,
        logging_steps=100,
        save_strategy="epoch",
        eval_steps=100,
        seed=SEED,
        report_to=["wandb"],
        fp16=True,
        bf16=False,
        remove_unused_columns=False,
        assistant_only_loss=False,
        completion_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        # tokenizer=tokenizer,
        peft_config=lora_config,
        args=training_args,
    )

    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)

    metrics_path = os.path.join(output_dir, "train_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(train_result.metrics, f)

    wandb.finish()


if __name__ == "__main__":
    model = "meta-llama/Llama-3.1-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    dataset = build_instruction_finetune_dataset(
        data_dir="data",
        tasks=TASKS + ["ner_bio"],
        tokenizer=tokenizer,
        split="train",
    )

    # Save dataset to jsonl file
    output_path = './finetuning/instruction_tune_dataset_train.jsonl'
    with open(output_path, 'w') as f:
        for example in dataset:
            f.write(json.dumps(example) + '\n')

    output_dir = './finetuning/sft_llama3_8b_instruction_tuned'
    data_file = './finetuning/instruction_tune_dataset_train.jsonl'
    sft_peft(model, data_file, output_dir)
