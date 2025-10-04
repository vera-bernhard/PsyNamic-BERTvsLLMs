#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: predic_zero_shot.py
Description: ...
Author: Vera Bernhard
"""
import os
import re
import json
import ast
from datetime import datetime

import pandas as pd
import spacy
import torch
from tqdm import tqdm
from typing import TextIO

from dotenv import load_dotenv
from huggingface_hub import login
from openai import OpenAI
from evaluation.helper import parse_file_name

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers.utils import logging

from typing import Literal

from prompts.build_prompts import (
    build_class_prompt,
    system_role_class,
    system_role_ner,
    build_ner_prompt,
    get_label2int,
    is_multilabel,
)

logging.set_verbosity_info()


def set_seed(seed: int = 42):
    """Set all seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


TASKS = [
    "Condition", "Data Collection", "Data Type", "Number of Participants", "Age of Participants", "Application Form",
    "Clinical Trial Phase",  "Outcomes", "Regimen", "Setting", "Study Control", "Study Purpose",
    "Substance Naivety", "Substances", "Sex of Participants", "Relevant", "Study Conclusion",
    "Study Type"
]

# TODO: Move it somewhere else so that it is not always called
# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
access_token = os.getenv("ACCESS_TOKEN")
login(access_token)

nlp = spacy.load("en_core_web_sm")
SEED = 42

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class LlamaModel():
    def __init__(self, model_name: str, use_gpu: bool = True, system_prompt: str = '', use_quant: bool = False):
        print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.model_name_short = model_name.split('/')[-1]
        self.system_prompt = system_prompt

        if self.use_gpu and torch.cuda.is_available():
            self.device_map = "auto"
            self.torch_dtype = torch.float16
        else:
            self.device_map = None
            self.torch_dtype = torch.float32
        print(f"Using device_map={self.device_map}, dtype={self.torch_dtype}")

        # Load tokenizer and set pad token
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        print(
            f"eos_token_id={self.tokenizer.eos_token_id}, pad_token_id={self.tokenizer.pad_token_id}")
        self.check_chat_template()

        if use_quant:
            # 4-bit quantization
            print("Loading model with 4-bit quantization")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="balanced",  # ensures that all GPUs are used
                torch_dtype=self.torch_dtype,
                quantization_config=bnb_config,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="balanced",
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

        print(f"Model loaded. Device map: {self.model.hf_device_map}")

    def check_chat_template(self):
        # Load chat template from backbone model if not present
        if self.tokenizer.chat_template is None:
            if self.model_name_short == 'Llama-2-70b-chat-hf':
                tokenizer2 = AutoTokenizer.from_pretrained(
                    'meta-llama/Llama-2-13b-chat-hf')
                self.tokenizer.chat_template = tokenizer2.chat_template
            elif self.model_name_short == 'MeLLaMA-13B-chat':
                tokenizer2 = AutoTokenizer.from_pretrained(
                    'meta-llama/Llama-2-13b-chat-hf')
                self.tokenizer.chat_template = tokenizer2.chat_template
            # no chat template for Med-LLaMA3-8B
            elif self.model_name_short == 'Med-LLaMA3-8B':
                return
            else:
                raise ValueError(
                    f"Model {self.model_name} does not have a chat template. Please provide one.")

    def build_prompt(self, prompt: str):
        # no chat template for Med-LLaMA3-8B
        if self.model_name_short == 'Med-LLaMA3-8B':
            return prompt
        message = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        prompt_with_template = self.tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True)
        return prompt_with_template

    def set_task(self, task: str):
        self.task = task
        if not 'ner' in task.lower():
            prompt_dir = os.path.join(SCRIPT_DIR, '..', 'prompts')
            file_path = os.path.join(
                prompt_dir, 'classification_description.json')

            with open(file_path, 'r', encoding='utf-8') as f:
                task_descriptions = json.load(f)
                output_format = json.dumps(
                    {key: 0 for key in task_descriptions[task]['Options'].keys(
                    )},
                    indent=4
                )
                # check how many tokens the output format has
                output_format_tokens = self.tokenizer(
                    output_format, return_tensors="pt").input_ids.shape[1]
                self.max_new_tokens = output_format_tokens + 50

        else:
            self.max_new_tokens = 256

    def predict(self, prompt_with_template: str, temperature: float = 0, do_sample: bool = False, top_p: float = 1.0) -> str:
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        inputs = self.tokenizer(prompt_with_template, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[1]
        max_ctx = self.model.config.max_position_embeddings
        if input_len > max_ctx:
            print(f"[WARN] Prompt length {input_len} exceeds context window ({max_ctx}). "
                  f"Input will be truncated.")

        # Determine max_new_tokens for NER tasks based on input length
        if self.task and 'ner' in self.task.lower():
            input_text = prompt_with_template.split('INPUT:')[-1]
            input_text = input_text.rstrip().rstrip('OUTPUT:').strip()
            self.max_new_tokens = len(input_text.split()) * 2

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "min_new_tokens": 1,
            "do_sample": do_sample,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "top_p": top_p,
            "temperature": None,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["generator"] = torch.Generator(
                device=self.model.device).manual_seed(SEED)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **generation_kwargs)

        # Debug decode
        full_text = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=False)
        # Slice off prompt
        gen_text = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        ).strip()
        if not gen_text:
            print("[WARN] Empty generation after slicing. Check EOS / prompt template.")

        return gen_text

    def batch_predict(self, prompts_with_template, temperature=0, do_sample=False, top_p=1.0):
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        if 'ner' in self.task.lower():
            max_tokens_per_prompt = [
                len(p.split('INPUT:')
                    [-1].rstrip().rstrip('OUTPUT:').strip().split()) * 2
                for p in prompts_with_template
            ]
            self.max_new_tokens = max(max_tokens_per_prompt)
            print(f"Max new tokens for NER batch: {self.max_new_tokens}")

        # Tokenize all prompts together (padding to max length in batch)
        inputs = self.tokenizer(
            prompts_with_template,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model.config.max_position_embeddings,
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        input_lens = [len(ids) for ids in inputs["input_ids"]]
        max_input_len = max(input_lens)
        max_ctx = self.model.config.max_position_embeddings
        if max_input_len > max_ctx:
            print(f"[WARN] Longest prompt length {max_input_len} exceeds context window ({max_ctx}). "
                  f"Inputs will be truncated.")

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "min_new_tokens": 1,
            "do_sample": do_sample,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "top_p": top_p,
            "temperature": None,
        }

        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["generator"] = torch.Generator(
                device=self.model.device).manual_seed(SEED)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **generation_kwargs)

        results = []
        for i, generated in enumerate(output_ids):
            full_text = self.tokenizer.decode(
                generated, skip_special_tokens=False)
            gen_text = self.tokenizer.decode(
                generated[inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()
            if not gen_text:
                print(f"[WARN] Empty generation for item {i}.")
            results.append(gen_text)

        return results


def gpt_prediction(prompt: str, model: str = "gpt-4o-mini", system_role: str = '') -> tuple[str, str]:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            seed=SEED  # not sure if this realyl works
        )
    except Exception as e:
        return f"OpenAI API call failed: {e}", model

    response_content = response.choices[0].message.content.strip()
    model_spec = response.model
    return response_content, model_spec


def make_class_predictions(
    tasks: list[str],
    model_name: str,
    limit: int = None,
    few_shot: int = 0,
    few_shot_strategy: Literal['selected', 'random'] = 'selected',
    batch_size: int = 8,
    skip_with_other_date: bool = True,
) -> None:
    if 'llama' in model_name.lower():
        model = LlamaModel(model_name=model_name,
                           system_prompt=system_role_class)
    else:
        model = None

    for task in tasks:
        task_lower = task.lower().replace(' ', '_')
        date = datetime.today().strftime('%d-%m-%d')

        if few_shot > 0:
            outfile = f"few_shot/{task_lower}/{task_lower}_{few_shot}shot_{few_shot_strategy}_{model_name.split('/')[-1]}_{date}.csv"
        else:
            outfile = f"zero_shot/{task_lower}/{task_lower}_{model_name.split('/')[-1]}_{date}.csv"
        os.makedirs(os.path.dirname(outfile), exist_ok=True)

        if skip_with_other_date:
            outfile_without_date = '_'.join(outfile.rstrip(
                '.csv').split('_')[:-1]).split('/')[-1]
            skip_task = False
            # if outfile already exists skip
            for existing_file in os.listdir(os.path.dirname(outfile)):
                if existing_file.startswith(outfile_without_date):
                    print(
                        f"Found existing prediction with this model: {existing_file}. Skipping...")
                    skip_task = True
                    break

            if skip_task:
                continue

        print(f"Processing task: {task}")
        file = os.path.join(os.path.dirname(__file__), '..',
                            'data', task_lower, 'test.csv')

        if not os.path.exists(file):
            raise FileNotFoundError(
                f"The file {file} does not exist. Check the task name.")
        df = pd.read_csv(file)

        if limit is not None:
            df = df.iloc[:limit]

        prompts = []
        predictions = []
        model_specs = []

        # Build all prompts first
        built_prompts = []
        for _, row in df.iterrows():
            prompt = build_class_prompt(
                row['id'], task, row['text'], few_shot=few_shot, few_shot_strategy=few_shot_strategy
            )
            if 'llama' in model_name.lower():
                model.set_task(task)
                built_prompts.append(model.build_prompt(prompt))
            else:
                built_prompts.append(prompt)

        # Predict in batches
        if 'llama' in model_name.lower():
            model_spec = model.model_name_short

            for i in tqdm(range(0, len(built_prompts), batch_size), desc=f"Predicting {task}"):
                batch = built_prompts[i:i + batch_size]
                if batch_size == 1:
                    prediction = model.predict(
                        batch[0], temperature=0, do_sample=False)
                    preds = [prediction]
                else:
                    preds = model.batch_predict(
                        batch, temperature=0, do_sample=False)
                prompts.extend(batch)
                predictions.extend(preds)
                model_specs.extend([model_spec] * len(batch))

        elif 'gpt' in model_name:
            for prompt in tqdm(built_prompts, desc=f"Predicting {task}"):
                prediction, model_spec = gpt_prediction(
                    prompt, model=model_name, system_role=system_role_class
                )
                prompts.append(prompt)
                predictions.append(prediction)
                model_specs.append(model_spec)

        # attach predictions back to df
        df['prompt'] = prompts
        df['prediction_text'] = predictions
        df['model'] = model_specs

        df_out = df[['id', 'text', 'prompt',
                     'prediction_text', 'model', 'labels']]
        print(f"Saving predictions to {outfile}")
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        df_out.to_csv(outfile, index=False, encoding='utf-8')


def make_ner_predictions(
    model_name: str,
    limit: int = None,
    few_shot: int = 0,
    few_shot_strategy: Literal['selected', 'random'] = 'selected',
    batch_size: int = 8,
    skip_with_other_date: bool = True,
) -> None:
    task = "ner_bio"
    file = os.path.join(os.path.dirname(__file__),
                        '..', 'data', task, 'test.csv')
    if not os.path.exists(file):
        raise FileNotFoundError(
            f"The file {file} does not exist. Check the task name.")

    date = datetime.today().strftime('%d-%m-%d')
    if few_shot > 0:
        outfile = f"few_shot/ner/ner_{few_shot}shot_{few_shot_strategy}_{model_name.split('/')[-1]}_{date}.csv"
    else:
        outfile = f"zero_shot/ner/ner_{model_name.split('/')[-1]}_{date}.csv"
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    # Skip prediction if a prediction file from another date already exists
    if skip_with_other_date:
        outfile_without_date = '_'.join(outfile.rstrip(
            '.csv').split('_')[:-1]).split('/')[-1]
        for existing_file in os.listdir(os.path.dirname(outfile)):
            if existing_file.startswith(outfile_without_date):
                print(
                    f"Found existing prediction with this model: {existing_file}. Skipping...")
                return

    if 'llama' in model_name.lower():
        model = LlamaModel(model_name=model_name,
                           system_prompt=system_role_ner)
        model.set_task(task)

    df = pd.read_csv(file)
    if limit is not None:
        df = df.iloc[:limit]

    prompts = []
    predictions = []
    model_specs = []

    # Build all prompts first
    built_prompts = []
    for _, row in df.iterrows():
        prompt = build_ner_prompt(
            row['id'], row['text'], few_shot=few_shot, few_shot_strategy=few_shot_strategy)
        if 'llama' in model_name.lower():
            built_prompts.append(model.build_prompt(prompt))
        else:
            built_prompts.append(prompt)

    if 'llama' in model_name.lower():
        model_spec = model.model_name_short
        for i in tqdm(range(0, len(built_prompts), batch_size), desc="Predicting (NER)"):
            batch = built_prompts[i:i + batch_size]
            if batch_size == 1:
                prediction = model.predict(
                    batch[0], temperature=0, do_sample=False)
                preds = [prediction]
            else:
                preds = model.batch_predict(
                    batch, temperature=0, do_sample=False)
            prompts.extend(batch)
            predictions.extend(preds)
            model_specs.extend([model_spec] * len(batch))

    elif 'gpt' in model_name:
        for prompt in tqdm(built_prompts, desc="Predicting (NER)"):
            prediction, model_spec = gpt_prediction(
                prompt, model=model_name, system_role=system_role_ner)
            prompts.append(prompt)
            predictions.append(prediction)
            model_specs.append(model_spec)

    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    df['prompt'] = prompts
    df['prediction_text'] = predictions
    df['model'] = model_specs

    df_out = df[['id', 'text', 'prompt', 'prediction_text', 'model', 'tokens']]
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    df_out.to_csv(outfile, index=False, encoding='utf-8')


def is_one_hot(string: str, length: int) -> bool:
    """Check if a string is a one-hot encoded list of given length.
    It recognizes cases like "0 0 1 0", "0,0,1,0", "[0, 0, 1, 0]", etc.
    """
    string = string.strip().lstrip("[").rstrip("]")

    if "," in string:
        parts = [x.strip() for x in string.split(",")]
    else:
        parts = [x.strip() for x in string.split()]

    if len(parts) != length:
        return False

    return all(part in {"0", "1"} for part in parts)


def is_first_line_one_hot(string: str, length: int) -> bool:
    """Check if the first line of a string is a one-hot encoded list of given length."""
    first_line = string.split('\n')[0].strip()
    return is_one_hot(first_line, length)


def parse_class_prediction(pred_text: str, label2int: dict, model: str) -> tuple[str, bool]:
    """Parse the prediction text from various generative llms into a one-hot encoded list of labels.

    Hand-crafted for the following models:
    - Llama-2
    - MeLLaMA
    - GPT-4o
    """
    # if pred text is nan
    if pd.isna(pred_text) or pred_text == '':
        raise ValueError("Prediction text is empty or NaN.")

    if model.startswith('Llama-2'):
        # Split at [/INST]
        parts = pred_text.split('[/INST]')
        pred_text = parts[-1].strip()

    elif model.startswith('MeLLaMA'):
        parts = pred_text.split('OUTPUT:')
        # if len(parts) != 2:
        #     breakpoint()
        #     raise ValueError(
        #         f'Prediction text does not contain "OUTPUT:": {pred_text}')
        pred_text = parts[-1].strip()

    elif model.startswith('gpt'):
        pred_text = pred_text.replace('\\n', '\n')

    elif model.startswith('Med-LLaMA3'):
        # remove INPUT and following text
        parts = pred_text.split('INPUT:')
        pred_text = parts[0].strip()

    faulty_but_parsable = False
    if '{' in pred_text:
        # Case 1: There is a prediction in dictionary format
        start = pred_text.index('{')
        try:
            end = pred_text.index('}')
        # There is no closing bracket
        except ValueError:
            # Add closing bracket at the end
            pred_text += '}'
            end = pred_text.index('}')
            faulty_but_parsable = True
        prediction_dict = pred_text[start:end+1]
        # Clean up the prediction dictionary so that is valid JSON
        prediction_dict = prediction_dict.replace('""', '"')
        prediction_dict = re.sub(r':\s*"\s*(?=[,}])', ': ""', prediction_dict)
        prediction_dict = json.loads(prediction_dict)

        # Check if there is empty predictions -> "" instead of 0 or 1
        if "" in prediction_dict.values():
            faulty_but_parsable = True
            # Check if there is at leas one 1
            if not any(value == 1 or value == '1' for value in prediction_dict.values()):
                print(f"Not parsable: {pred_text}")
                faulty_but_parsable = False
                raise ValueError(f'Could not parse prediction: {pred_text}')
            else:
                # replace empty predictions with 0
                for key in prediction_dict.keys():
                    if prediction_dict[key] == "":
                        prediction_dict[key] = 0

        # Create one-hot encoding, according to label2int
        onehot_list = [0] * len(label2int)
        for label, value in prediction_dict.items():
            # Sanity check: label exists in label2int
            if label not in label2int:
                label_syn = check_label_synonyms(label)
                if label_syn is not None:
                    label = label_syn
                else:
                    raise ValueError(
                        f'Label {label} not found in label2int mapping.')
            pos = label2int[label]
            onehot_list[pos] = int(value)
        return str(onehot_list), faulty_but_parsable

    elif pred_text in label2int.keys():
        faulty_but_parsable = True
        # Case 2: There is a prediction in string format, e.g. 'Randomized-controlled trial (RCT)'
        onehot_list = [0] * len(label2int)
        pos = label2int[pred_text]
        onehot_list[pos] = 1
        return str(onehot_list), faulty_but_parsable

    elif is_one_hot(pred_text, len(label2int)):
        faulty_but_parsable = True
        # Case 4: There is a prediction in one-hot format with valid length, e.g. '0, 1, 0, 0, 0, 0, 0, 0, 0'
        if ',' in pred_text:
            onehot_list = [int(x.strip()) for x in pred_text.split(",")]
        else:
            onehot_list = [int(x.strip()) for x in pred_text.split()]
        return str(onehot_list), faulty_but_parsable

    elif is_first_line_one_hot(pred_text, len(label2int)):
        faulty_but_parsable = True
        # Case 5: There is a prediction in one-hot format with valid length in the first line, e.g. '0, 1, 0, 0, 0, 0, 0, 0, 0\nSome explanation'
        first_line = pred_text.split('\n')[0].strip()
        if ',' in first_line:
            onehot_list = [int(x.strip()) for x in first_line.split(",")]
        else:
            onehot_list = [int(x.strip()) for x in first_line.split()]
        return str(onehot_list), faulty_but_parsable

    elif ':' in pred_text:
        if pred_text.count(':') > 1:
            preds = pred_text.split('\n')
            # remove empty lines
            preds = [p for p in preds if p.strip() != '']
            onehot_list = [0] * len(label2int)
            for p in preds:
                label = p.split(':')[0].strip()
                value = p.split(':')[1].strip()
                if label not in label2int:
                    label_syn = check_label_synonyms(label)
                    if label_syn is not None:
                        label = label_syn
                    else:
                        raise ValueError(
                            f'Label {label} not found in label2int mapping.')
                pos = label2int[label]
                onehot_list[pos] = int(value)
            faulty_but_parsable = True
            return str(onehot_list), faulty_but_parsable

        faulty_but_parsable = True
        # Case 3: There is a prediction in string format with a score, e.g. 'Randomized-controlled trial (RCT): 1
        onehot_list = [0] * len(label2int)
        label = pred_text.split(':')[0].strip()
        pos = label2int[label]
        onehot_list[pos] = 1
        return str(onehot_list), faulty_but_parsable

    else:
        raise ValueError(f'Could not parse prediction: {pred_text}')


def check_label_synonyms(label: str) -> str:
    synonyms = {
        "Alzheimer's disease": "Alzheimer’s disease",
        "Alzheimer\\u2019s disease": "Alzheimer’s disease",
        ">=1000": "≥1000",
        "Substance-Naive Participants": "Substance-Naïve Participants",
        "Substance-non-naive participants": "Substance-non-naïve participants"
    }
    try:
        return synonyms[label]
    except KeyError:
        return None


def parse_class_predictions(pred_file: str, task: str, reparse: bool = False, log_file: TextIO = None) -> dict:
    """Parse the predictions from a file and add a new column with one-hot encoded labels."""
    # Check if already parsed
    df_check = pd.read_csv(pred_file)
    if 'pred_labels' in df_check.columns and not reparse:
        print(
            f"The file {pred_file} already contains the column 'pred_labels'. Skipping parsing.")
        return {}

    if log_file is not None:
        log_file.write(
            f"Parsing predictions in file {pred_file} for task {task}\n")
        log_file.flush()

    label2int = get_label2int(task)
    multilabel = is_multilabel(task)

    # Check if file and column exist
    if not os.path.exists(pred_file):
        raise FileNotFoundError(
            f"The file {pred_file} does not exist. Check the task name.")

    df = pd.read_csv(pred_file)
    if 'prediction_text' not in df.columns:
        raise ValueError(
            f"The file {pred_file} does not contain the column 'prediction_text'. Check the file format.")

    # Parse model name from the file name --> model name before date dd-mm-dd.csv
    model = os.path.basename(pred_file).split('_')[-2]

    # add new column 'pred_labels' containing the one-hot encoded labels
    nr_non_parsable = 0
    nr_faulty_parsable = 0
    # change dtype of labels column to string
    # add nan
    df['pred_labels'] = None
    df['labels'] = df['labels'].astype(str)
    for i, row in df.iterrows():
        if not multilabel:
            # Check if it is one-hot encoded already
            if is_one_hot(row['labels'], len(label2int)):
                pass
            else:
                # Convert the true label to one-hot encoding
                true_label = int(row['labels'])
                true_onehot = [0] * len(label2int)
                true_onehot[true_label] = 1
                df.at[i, 'labels'] = str(true_onehot)
        try:
            pred_labels, faulty_but_parsable = parse_class_prediction(
                row['prediction_text'], label2int, model)
            df.at[i, 'pred_labels'] = pred_labels
            if faulty_but_parsable:
                nr_faulty_parsable += 1
        except Exception as e:
            if log_file is not None:
                log_file.write(
                    f"\tError parsing prediction for row with id {row['id']} : {e}\n")
                log_file.flush()
            else:
                print(
                    f"Error parsing prediction for row with id {row['id']} : {e}")
            nr_non_parsable += 1
            continue

    df.to_csv(pred_file, index=False, encoding='utf-8')
    stats = {
        'nr_faulty_parsable': nr_faulty_parsable,
        'nr_non_parsable': nr_non_parsable,
    }
    return stats


def parse_ner_prediction_bio(pred: str, tokens: list[str], text: str, log_file: TextIO) -> list[str]:
    """ Convert model prediction text with <span> tags into a BIO sequence.

    The prediction contains entities like:
        <span class="application-area">depression</span>
        <span class="dosage">50 mg</span>

    Steps:
    1. Clean the prediction text depending on the model (Llama-2 / MeLLaMA).
    2. Extract spans via regex: (entity text, entity type).
    3. Align each span with tokens:
       - First token → B-<EntityLabel>, rest → I-<EntityLabel>.
       - Restore from backup if partial match fails.
       - Fallback: search span text directly in tokens.
    4. Warn if number of B- labels doesn’t match number of spans.
    """
    # if text is nan
    if pd.isna(text) or text == '':
        return ['O'] * len(tokens)

    #TODO: Redunancy with function to extract entities
    ner_labels = {
        'application-area': 'Application area',
        'dosage': 'Dosage',
    }
    text = text.replace('\xa0', ' ')

    bio_tokens = ['O'] * len(tokens)
    token_pointer = 0

    # Don't do this twice
    spans = parse_ner_prediction_entities(pred)

    # Remove all spans with entity type not in ner_labels
    original_span_count = len(spans)
    removed_spans = [(e, t) for e, t in spans if t not in ner_labels]
    spans = [(e, t) for e, t in spans if t in ner_labels]
    removed_count = original_span_count - len(spans)
    if removed_count > 0:
        log_file.write(f"Removed {removed_count} spans with unknown entity types:\n")
        for e, t in removed_spans:
            log_file.write(f"  Removed entity: '{e}' (type: {t})\n")

    # Remove any spans that there string don't appear in the text at all, case insensitive
    original_span_count = len(spans)
    removed_spans = [(e, t) for e, t in spans if e.lower() not in text.lower()]
    spans = [(e, t) for e, t in spans if e.lower() in text.lower()]
    removed_count = original_span_count - len(spans)
    if removed_count > 0:
        log_file.write(f"Removed {removed_count} spans that don't appear in the text:\n")
        for e, t in removed_spans:
            log_file.write(f"  Removed entity: '{e}' (type: {t})\n")

    # Iterate through the spans and in parallel through the tokens to find matches
    # - assuming that the spans are in the same order as the tokens
    # - assuming that the spans are not overlapping (which is the case for the PsyNamic Scale)
    for e, t in spans:
        if t not in ner_labels:
            log_file.write(
                f'Warning: Entity {t} not found in ner_labels mapping. Available labels: {ner_labels.keys()}\n')
            continue
        e_in_bio = False  # Keep track if entity is matched
        temp_e = e.lower()  # Keep track what of the entity is still to be matched
        token_start = False  # Keep track of whether B- or I- token need to be set
        # Backup bio_tokens to reset if the beginning of the entity seemed a match but later turned out not to be
        backup_bio_tokens = bio_tokens.copy()

        # Iterate through the tokens to find matches
        for i in range(token_pointer, len(tokens)):
            if tokens[i] == '\n':
                continue
            token = tokens[i].strip().lower().rstrip('.^')
            if not token_start:
                if temp_e.startswith(token):
                    bio_tokens[i] = f'B-{ner_labels[t]}'
                    token_start = True
                    temp_e = temp_e.lstrip(token).strip()
                    e_in_bio = True
            else:
                if temp_e.startswith(token):
                    bio_tokens[i] = f'I-{ner_labels[t]}'
                    temp_e = temp_e.lstrip(token).strip()

                elif temp_e.startswith(clean_token(token)):
                    bio_tokens[i] = f'I-{ner_labels[t]}'
                    temp_e = temp_e.lstrip(clean_token(token)).strip()

                else:
                    # Case 1: all of the entity is matched and only then save in bio_tokens
                    if temp_e == '':
                        token_start = False
                        token_pointer = i + 1
                        break  # Move to next entity

                    # Case 2: Only part of the entity is matched --> entity must be later in the text or be faulty
                    else:
                        e_in_bio = False
                        bio_tokens = backup_bio_tokens.copy()
                        token_start = False
                        temp_e = e.lower()

        # In cases, of very messy prediction, where
        # - the order of spans in predictions is not the same as they appear in tokens
        # - the prediction text is not the same as the tokens but spans are still in text
        if not e_in_bio:
            if e in ' '.join(tokens):
                ids = find_phrase_indices(tokens, e)
                # set first token to B- and the rest to I-
                # Check that bio_token at ids are still 'O'
                if ids:
                    # Only set Bio if nothing previously set
                    if all(bio_tokens[i] == 'O' for i in ids):
                        bio_tokens[ids[0]] = f'B-{ner_labels[t]}'
                        for j in ids[1:]:
                            bio_tokens[j] = f'I-{ner_labels[t]}'

    # Check if there is as many entities in bio_token as in spans
    num_entities = sum(1 for label in bio_tokens if label.startswith('B-'))
    if num_entities > len(spans):
        log_file.write(
            f"Warning: Number of entities in bio_token ({num_entities}) is greater than number of spans ({len(spans)}).\n")

    if num_entities != len(spans):
        log_file.write(
            f"Warning: Number of entities in bio_token ({num_entities}) does not match number of spans ({len(spans)}).\n")
    log_file.flush()
    return bio_tokens


def parse_ner_prediction_entities(pred: str) -> list[tuple[str, str]]:
    spans = re.findall(r'<span class="(.*?)">(.*?)</span>', pred)
    spans = [(e, t) for t, e in spans]
    spans = [(e.strip(), t.strip('"')) for e, t in spans]
    return spans


def parse_ner_predictions(pred_file: str, reparse: bool = False, log_file: TextIO = None) -> None:
    if log_file is not None:
        log_file.write(
            f"Parsing predictions in file {pred_file} for NER\n")
        log_file.flush()
    if not os.path.exists(pred_file):
        raise FileNotFoundError(
            f"The file {pred_file} does not exist. Check the task name.")
    df = pd.read_csv(pred_file)

    if 'pred_labels' in df.columns and 'entities' in df.columns and not reparse:
        log_file.write(
        f"The file {pred_file} already contains the columns 'pred_labels' and 'entities'. Skipping parsing.\n")
        log_file.flush()
      
        return

    # Check if column 'prediction_text' exists
    if 'prediction_text' not in df.columns:
        raise ValueError(
            f"The file {pred_file} does not contain the column 'prediction_text'. Check the file format.")

    model = parse_file_name(pred_file, 'model')
    for i, row in df.iterrows():
        bio = parse_ner_prediction_bio(
            row['prediction_text'], ast.literal_eval(row['tokens']), row['text'], log_file)
        entities = parse_ner_prediction_entities(row['prediction_text'])
        df.at[i, 'pred_labels'] = str(bio)
        df.at[i, 'pred_entities'] = str(entities)

    df.to_csv(pred_file, index=False, encoding='utf-8')


def basic_tokenizer(text: str) -> list[str]:
    text = text.strip()
    doc = nlp(text)
    tokenized = [token.text for token in doc]
    return tokenized


def find_phrase_indices(tokens: list[str], phrase: str) -> list[int]:
    tokens_lower = [t.lower() for t in tokens]
    phrase_tokens = [t.lower() for t in phrase.split()]
    phrase_len = len(phrase_tokens)

    for i in range(len(tokens_lower) - phrase_len + 1):
        if tokens_lower[i:i + phrase_len] == phrase_tokens:
            return list(range(i, i + phrase_len))

    return []


def clean_token(token: str) -> str:
    """Clean the token by removing newlines and trailing punctuation."""
    token = token.strip().replace('\n', '')
    if token.endswith('.^'):
        token = token[:-2].strip()
    return token


def main():

    uzh_models = [
        # "/scratch/vebern/models/Llama-2-13-chat-hf",
        # "/scratch/vebern/models/Llama-2-70-chat-hf",
        # "/data/vebern/ma-models/MeLLaMA-13B-chat",
        # "/data/vebern/ma-models/MeLLaMA-70B-chat",
    ]
    models = [
        # "gpt-4o-mini",
        # "gpt-4o-2024-08-06",
        'meta-llama/Llama-2-13b-chat-hf',
        '/storage/homefs/vb25l522/me-llama/MeLLaMA-13B-chat',
        'meta-llama/Meta-Llama-3-8B-Instruct',
        'YBXL/Med-LLaMA3-8B',
        "meta-llama/Llama-3.1-8B-Instruct",

    ]
    big_models = [
        "/data/vebern/ma-models/MeLLaMA-70B-chat",
        "meta-llama/Llama-2-70b-hf",
        "meta-llama/Llama-3.3-70B-Instruct",
    ]

    # Zero-Shot: Classification
    for model_name in models:
        make_class_predictions(
            tasks=TASKS, model_name=model_name, batch_size=8, few_shot=0, skip_with_other_date=True)
        make_ner_predictions(model_name=model_name,
                             batch_size=8, few_shot=0, skip_with_other_date=True)

    # Few-Shot: Classification & NER
    for i in [1, 3, 5]:
        for model_name in models:

            make_class_predictions(
                tasks=TASKS, model_name=model_name, batch_size=8, few_shot=i, few_shot_strategy='selected', skip_with_other_date=True)

            make_ner_predictions(
                model_name=model_name, batch_size=8, few_shot=i, few_shot_strategy='selected', skip_with_other_date=True)


if __name__ == "__main__":
    main()
