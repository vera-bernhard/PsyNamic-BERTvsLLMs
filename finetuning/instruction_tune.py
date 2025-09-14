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

# Prompt completion dataset
#  {"prompt": "The sky is",
#  "completion": " blue."}

def build_instruction_finetune_dataset(
    data_dir: str,
    tasks: list[str],
    split = 'train'
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
                output_json = build_json_labels(row['labels'], int2label, options)

                dataset.append({
                    "prompt": message,
                    "completion": [{"role": "assistant", "content": output_json}]
                })

    random.shuffle(dataset)
    return dataset


   
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