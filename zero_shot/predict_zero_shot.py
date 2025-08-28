#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: predic_zero_shot.py
Description: ...
Author: Vera Bernhard
"""

from huggingface_hub import login
import os
import re
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from datetime import datetime
from prompts.build_prompts import build_class_prompt, system_role_class, system_role_ner, build_ner_prompt, build_llama2_prompt
from openai import OpenAI
from typing import Literal
import pandas as pd
import json
import spacy
import ast
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import deepspeed
import bitsandbytes as bnb
from transformers import BitsAndBytesConfig
from transformers.utils import logging
from tqdm import tqdm
from typing import Union
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


class LlamaModel():
    def __init__(self, model_name: str, use_gpu: bool = True, is_ner: bool = False, system_prompt: str = ''):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.model_name_short = model_name.split('/')[-1]
        self.is_ner = is_ner
        self.system_prompt = system_prompt

        if self.use_gpu:
            self.device_map = "auto"
            self.torch_dtype = torch.float16
            self.device = torch.device("cuda")
        else:
            self.device_map = None
            self.torch_dtype = torch.float32
            self.device = torch.device("cpu")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if "70b" in self.model_name.lower():
            # 4-bit quantization
            print("Loading a 70B model with 4-bit quantization")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device_map,
                torch_dtype=self.torch_dtype,
                quantization_config=bnb_config,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device_map,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
            )
            self.model.to(self.device)


    def build_prompt(self, prompt: str):
        try: 
            message = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            prompt_with_template = self.tokenizer.apply_chat_template(message, tokenize=False)
        except ValueError:
            if 'llama2' in self.model_name.lower():
                prompt_with_template = build_llama2_prompt(prompt, self.system_prompt)
            elif 'mellama' in self.model_name.lower():
                prompt_with_template = prompt
            else:
                raise ValueError("Prompting not supported for this model.")

        return prompt_with_template

    def predict(self, prompt_with_template: str, temperature: float = 0, do_sample: bool = False, top_p: float = 1.0) -> str:
        # Set seed for reproducibility
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        inputs = self.tokenizer(prompt_with_template, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        generation_kwargs = {
            "max_new_tokens": 500,
            "do_sample": do_sample,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "top_p": top_p,
            "temperature": None, # unset temperature if do_sample is False, cause the argument is not considered
        }

        if do_sample:
            generation_kwargs["temperature"] = temperature
            generator = torch.Generator(device=self.device).manual_seed(SEED)
            generation_kwargs["generator"] = generator

        with torch.no_grad():
            generation_output = self.model.generate(
                **inputs, **generation_kwargs)

        output_text = self.tokenizer.decode(
            generation_output[0][inputs["input_ids"].shape[-1]:],  # skip prompt
            skip_special_tokens=True)

        # print(output_text)
        
        return output_text

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



def make_class_predictions(tasks: [str], model_name: str, limit: int = None, few_shot: int = 0):
    if 'llama' in model_name.lower():
        model = LlamaModel(model_name=model_name, system_prompt=system_role_class)

    for task in tasks:
        task_lower = task.lower().replace(' ', '_')
        date = datetime.today().strftime('%d-%m-%d')
        outfile = f"zero_shot/{task_lower}/{task_lower}_{model_name.split('/')[-1]}_{date}.csv"
        os.makedirs(os.path.dirname(outfile), exist_ok=True)

        print(f"Processing task: {task}")
        task_lower = task.lower().replace(' ', '_')
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

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Predicting {task}"):
            prompt = build_class_prompt(row['id'],task, row['text'], few_shot)

            # Llama chat
            if 'llama' in model_name.lower():
                prompt_with_template = model.build_prompt(prompt)
                model_spec = model.model_name_short
                prediction = model.predict(prompt_with_template, temperature=0, do_sample=False)

            elif 'gpt' in model_name:
                prediction, model_spec = gpt_prediction(
                    prompt, model=model_name, system_role=system_role_class)

            prompts.append(prompt)
            predictions.append(prediction)
            model_specs.append(model_spec)

        df['prompt'] = prompts
        df['prediction_text'] = predictions
        df['model'] = model_specs

        df_out = df[['id', 'text', 'prompt', 'prediction_text',
                    'model', 'labels']]
        print(f"Saving predictions to {outfile}")
        df_out.to_csv(outfile, index=False, encoding='utf-8')


def make_ner_predictions(model_name: str, outfile: str, limit: int = None, few_shot: int = 0):
    task = "ner_bio"
    file = os.path.join(os.path.dirname(__file__),
                        '..', 'data', task, 'test.csv')

    if not os.path.exists(file):
        raise FileNotFoundError(
            f"The file {file} does not exist. Check the task name.")

    df = pd.read_csv(file)

    if limit is not None:
        df = df.iloc[:limit]

    prompts = []
    predictions = []
    model_specs = []

    if 'llama' in model_name.lower():
        model = LlamaModel(model_name=model_name, is_ner=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting (NER)"):
        prompt = build_ner_prompt(row['id'], row['text'], few_shot=few_shot)

        if 'llama-2' in model_name.lower():
            prompt = build_llama2_prompt(prompt, system_role_ner)

        if 'llama' in model_name.lower():
            model_spec = model.model_name_short
            prediction = model.predict(prompt, temperature=0, do_sample=False)

        elif 'gpt' in model_name:
            prediction, model_spec = gpt_prediction(
                prompt, model=model_name, system_role=system_role_ner)

        else:
            raise ValueError(f"Unsupported model type: {model_name}")

        prompts.append(prompt)
        predictions.append(prediction)
        model_specs.append(model_spec)

    df['prompt'] = prompts
    df['prediction_text'] = predictions
    df['model'] = model_specs

    df_out = df[['id', 'text', 'prompt', 'prediction_text', 'model', 'tokens']]
    df_out.to_csv(outfile, index=False, encoding='utf-8')


def parse_class_prediction(pred_text: str, label2int: dict, model: str) -> str:
    """Parse the prediction text from various generative llms into a one-hot encoded list of labels.

    Hand-crafted for the following models:
    - Llama-2
    - MeLLaMA
    - GPT-4o
    """

    if model.startswith('Llama-2'):
        # Split at [/INST]
        parts = pred_text.split('[/INST]')
        pred_text = parts[-1].strip()

    elif model.startswith('MeLLaMA'):
        parts = pred_text.split('OUTPUT:')
        if len(parts) != 2:
            raise ValueError(
                f'Prediction text does not contain "OUTPUT:": {pred_text}')
        pred_text = parts[-1].strip()

    elif model.startswith('gpt'):
        pred_text = pred_text.replace('\\n', '\n')

    if '{' in pred_text:
        # Case 1: There is a prediction in dictionary format
        start = pred_text.index('{')
        end = pred_text.index('}')
        prediction_dict = pred_text[start:end+1]
        # Clean up the prediction dictionary so that is valid JSON
        prediction_dict = prediction_dict.replace('""', '"')
        prediction_dict = re.sub(r':\s*"\s*(?=[,}])', ': ""', prediction_dict)

        prediction_dict = json.loads(prediction_dict)

        # Check if there is empty predictions -> "" instead of 0 or 1
        if "" in prediction_dict.values():
            # Check if there is at leas one 1
            if not any(value == '1' for value in prediction_dict.values()):
                print(f"Not parsable: {pred_text}")
                return None
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
                raise ValueError(
                    f'Label {label} not found in label2int mapping.')
            pos = label2int[label]
            onehot_list[pos] = int(value)

    elif pred_text in label2int.keys():
        # Case 2: There is a prediction in string format, e.g. 'Randomized-controlled trial (RCT)'
        onehot_list = [0] * len(label2int)
        pos = label2int[pred_text]
        onehot_list[pos] = 1
        return str(onehot_list)

    elif ':' in pred_text:
        # Case 3: There is a prediction in string format with a score, e.g. 'Randomized-controlled trial (RCT): 1
        onehot_list = [0] * len(label2int)
        label = pred_text.split(':')[0].strip()
        pos = label2int[label]
        onehot_list[pos] = 1
        return str(onehot_list)
    else:
        return None

    return str(onehot_list)


def parse_class_predictions(pred_file: str, task: str) -> None:
    """Parse the predictions from a file and add a new column with one-hot encoded labels."""

    label2int = get_label2int(task)

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
    for i, row in df.iterrows():
        try:
            if i == 748:
                pass
            # write new column 'pred_labels' with parsed prediction
            df.at[i, 'pred_labels'] = parse_class_prediction(
                row['prediction_text'], label2int, model)
        except Exception as e:
            print(
                f"Error parsing prediction for row with id {row['id']} : {e}")
            continue

    df.to_csv(pred_file, index=False, encoding='utf-8')


def parse_ner_prediction(pred: str, tokens: list[str], model: str) -> list[str]:
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

    ner_labels = {
        'application-area': 'Application area',
        'dosage': 'Dosage',
    }

    # TODO: Deduplicate with class parse predictions
    if 'Llama-2' in model:
        parts = pred.split('[/INST]')
        pred = parts[-1].strip()

    elif 'MeLLaMA' in model:
        parts = pred.split('OUTPUT:')
        if len(parts) != 2:
            raise ValueError(
                f'Prediction text does not contain "OUTPUT:": {pred}')
        pred = parts[-1].strip()

    bio_tokens = ['O'] * len(tokens)
    token_pointer = 0

    # Extract spans from the prediction text
    spans = re.findall(r'<span class="(.*?)">(.*?)</span>', pred)
    spans = [(e, t) for t, e in spans]
    spans = [(e.strip(), t.strip('"')) for e, t in spans]

    # Iterate through the spans and in parallel through the tokens to find matches
    # - assuming that the spans are in the same order as the tokens
    # - assuming that the spans are not overlapping (which is the case for the PsyNamic Scale)
    for e, t in spans:
        if t not in ner_labels:
            print(
                f'Warning: Entity {t} not found in ner_labels mapping. Available labels: {ner_labels.keys()}')
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
            token = tokens[i].strip().lower()
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
        print(
            f"Warning: Number of entities in bio_token ({num_entities}) is greater than number of spans ({len(spans)}).")

    if num_entities != len(spans):
        print(
            f"Warning: Number of entities in bio_token ({num_entities}) does not match number of spans ({len(spans)}).")

    return bio_tokens


def parse_ner_predictions(file: str, output: Literal['entities', 'tokens']) -> None:
    # Check if file exists
    if not os.path.exists(file):
        raise FileNotFoundError(
            f"The file {file} does not exist. Check the task name.")

    # Check if column 'prediction_text' exists
    df = pd.read_csv(file)
    if 'prediction_text' not in df.columns:
        raise ValueError(
            f"The file {file} does not contain the column 'prediction_text'. Check the file format.")

    # Parse model name from the file name --> model name before date dd-mm-dd.csv
    model = os.path.basename(file).split('_')[-2]

    for i, row in df.iterrows():
        bio = parse_ner_prediction(
            row['prediction_text'], ast.literal_eval(row['tokens']), model)
        df.at[i, 'pred_labels'] = str(bio)

    # Remove entities column if it exists
    if 'entities' in df.columns:
        df = df.drop(columns=['entities'])

    df.to_csv(file, index=False, encoding='utf-8')


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


def get_label2int(task: str) -> dict:
    task_lower = task.lower().replace(' ', '_')
    file = os.path.join(os.path.dirname(__file__), '..',
                        'data', task_lower, 'meta.json')

    if not os.path.exists(file):
        raise FileNotFoundError(
            f"The file {file} does not exist. Check the task name.")

    with open(file, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    int2label = meta.get('Int_to_label', {})
    label2int = {v: int(k) for k, v in int2label.items()}
    return label2int


def add_tokens_nertags(bioner_path: str):
    test_path = '/home/vera/Documents/Uni/Master/Master_Thesis2.0/PsyNamic-Scale/data/ner_bio/test.csv'
    df_full = pd.read_csv(test_path)
    df_bioner = pd.read_csv(bioner_path)
    # add ner_tags if id matches
    for i, row in df_bioner.iterrows():
        id = row['id']
        if id in df_full['id'].values:
            matching_row = df_full[df_full['id'] == id]
            if not matching_row.empty:
                df_bioner.at[i, 'ner_tags'] = matching_row['ner_tags'].values[0]
    df_bioner.to_csv(bioner_path, index=False, encoding='utf-8')


def main():

    models = [
        # "/scratch/vebern/models/Llama-2-13-chat-hf",
        # "/scratch/vebern/models/Llama-2-70-chat-hf",
        # "/data/vebern/ma-models/MeLLaMA-13B-chat",
        # "/data/vebern/ma-models/MeLLaMA-70B-chat",
        # "gpt-4o-mini"
        'meta-llama/Llama-2-13b-chat-hf',
        '/storage/homefs/vb25l522/me-llama/MeLLaMA-13B-chat',
        'meta-llama/Llama-2-70b-chat-hf',
        '/storage/homefs/vb25l522/me-llama/MeLLaMA-70B-chat',
        # "YBXL/Med-LLaMA3-8B"
    ]

    # Zero-Shot: Classification
    for model_name in models:
        if model_name == 'meta-llama/Llama-2-13b-chat-hf':
            make_class_predictions(tasks=TASKS[8:-1], model_name=model_name, few_shot=0)
        elif model_name == '/storage/homefs/vb25l522/me-llama/MeLLaMA-13B-chat':
            make_class_predictions(tasks=TASKS[:-1], model_name=model_name, few_shot=0)
        else:
            make_class_predictions(tasks=TASKS, model_name=model_name, few_shot=0)

    # Zero-Shot: NER
    # for model_name in models:
    #     outfile_ner = f"zero_shot/ner_{model_name.split('/')[-1]}_{date}.csv"
    #     make_ner_predictions(model_name, outfile_ner)

    # for file in os.listdir('zero_shot/ner'):
    #     if not file.endswith('.csv'):
    #         continue
    #     print(f"Processing file: {file}")
    #     file_path = os.path.join('zero_shot/ner', file)
    #     add_tokens_nertags(file_path)

if __name__ == "__main__":
    main()
