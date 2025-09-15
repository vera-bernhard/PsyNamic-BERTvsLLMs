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

from zero_shot.predict_zero_shot import TASKS

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
    split='train'
) -> list[dict[str, str]]:
    """
    Aggregate all splits of all classification tasks into a single shuffled instruction tuning dataset.
    Each example is a dict with 'prompt' and 'completion' keys.
    """
    random.seed(SEED)
    dataset = []

    for task in tasks:
        task_lower = task.lower().replace(' ', '_')
        split_path = os.path.join(data_dir, task_lower, f'{split}.csv')
        if not os.path.exists(split_path):
            print(f"Warning: {split_path} does not exist, skipping.")
            continue
        df = pd.read_csv(split_path)
        if 'ner' in task_lower:
            for _, row in df.iterrows():
                prompt = build_ner_prompt(
                    row['id'],
                    row['text'],
                    few_shot=0,
                    few_shot_strategy="selected",
                )
                system_prompt = system_role_ner

                # Split at last OUTPUT
                message = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
                tokens = ast.literal_eval(row['tokens'])
                labels = ast.literal_eval(row['ner_tags'])
                text = row['text']

                output = markup_entities(tokens, text, labels)

                dataset.append({
                    "prompt": message,
                    "completion": [{"role": "assistant", "content": output}]
                })

        else:
            int2label = get_int2label(task)
            for _, row in df.iterrows():
                prompt = build_class_prompt(
                    row['id'],
                    task,
                    row['text'],
                    few_shot=0,
                    few_shot_strategy="selected",
                )
                if 'ner' in task_lower:
                    system_prompt = system_role_ner
                else:
                    system_prompt = system_role_class

                # Split at last OUTPUT
                message = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]

                options = get_class_options(task)
                output_json = build_json_labels(
                    row['labels'], int2label, options)

                dataset.append({
                    "prompt": message,
                    "completion": [{"role": "assistant", "content": output_json}]
                })

    random.shuffle(dataset)
    return dataset


def sft_peft(model_name: str, data_file: str, output_dir: str, resume_from_checkpoint: Optional[str] = None,):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files={"train": data_file})
    train_dataset = dataset["train"]

    bnb_kwargs = {}

    # Use 4-bit quantization for large models
    if '70b' in model_name:
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
        dataset_text_field=None,
        max_seq_length=tokenizer.model_max_length,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        num_train_epochs=3,
        logging_steps=20,
        save_strategy="steps",
        eval_steps=20,
        seed=SEED,
        report_to=["wandb"],
        fp16=not '70b' in model_name,
        bf16=False,
        remove_unused_columns=False,
        assistant_only_loss=True,
        completion_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
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
    dataset = build_instruction_finetune_dataset(
        data_dir='data',
        tasks=TASKS + ['ner_bio'],
        split='train'
    )

    # Save dataset to jsonl file
    output_path = './finetuning/instruction_tune_dataset_train.jsonl'
    with open(output_path, 'w') as f:
        for example in dataset:
            f.write(json.dumps(example) + '\n')
