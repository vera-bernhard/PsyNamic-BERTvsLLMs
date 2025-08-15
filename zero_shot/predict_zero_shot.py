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
from prompts.build_prompts import build_class_prompt, system_role_class, system_role_ner, build_ner_prompt
from openai import OpenAI
import pandas as pd
import json
import spacy
import ast
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import deepspeed
from transformers.utils import logging
logging.set_verbosity_info()


# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
access_token = os.getenv("ACCESS_TOKEN")
login(access_token)

nlp = spacy.load("en_core_web_sm")
SEED = 42


def set_seed(seed: int = 42):
    """Set all seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


TASKS = [
    "Data Collection", "Data Type", "Number of Participants", "Age of Participants", "Application Form",
    "Clinical Trial Phase", "Condition", "Outcomes", "Regimen", "Setting", "Study Control", "Study Purpose",
    "Substance Naivety", "Substances", "Sex of Participants", "Study Conclusion", "Study Type", "Relevant"
]

class LlamaModel():
    def __init__(self, model_name: str, use_gpu: bool = False, distributed: bool = False):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.distributed = distributed
        self.model_name_short = model_name.split('/')[-1]

        # Set device and dtype
        if self.distributed and not self.use_gpu:
            raise ValueError(
                "DeepSpeed distributed inference requires GPU (use_gpu=True).")

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

        # Load model and optionally wrap with DeepSpeed
        if self.distributed:
            # Load model first on CPU to avoid OOM
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model = deepspeed.init_inference(
                model,
                mp_size=torch.cuda.device_count(),
                dtype=torch.float16 if self.use_gpu else torch.float32,
                replace_method="auto",
                replace_with_kernel_inject=True,
            )
            # DeepSpeed models are already on GPU, device for inputs only
            self.device = torch.device("cuda")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device_map,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,

            )
            self.model.to(self.device)

    def predict(self, prompt: str, temperature: float = 0, do_sample: bool = False, top_p: float = 1.0) -> str:
        # Set seed for reproducibility
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        # Tokenize inputs and move to device
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        generation_kwargs = {
            "max_new_tokens": 500,
            "do_sample": do_sample,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "temperature": temperature,
            "top_p": top_p,
        }

        if do_sample:
            generator = torch.Generator(device=self.device).manual_seed(SEED)
            generation_kwargs["generator"] = generator

        with torch.no_grad():
            generation_output = self.model.generate(
                **inputs, **generation_kwargs)

        output_text = self.tokenizer.decode(
            generation_output[0], skip_special_tokens=True)
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


def make_class_predictions(task: str, model_name: str, outfile: str, limit: int = None):
    task_lower = task.lower().replace(' ', '_')
    file = os.path.join(os.path.dirname(__file__), '..',
                        'data', task_lower, 'test.csv')

    if not os.path.exists(file):
        raise FileNotFoundError(
            f"The file {file} does not exist. Check the task name.")
    df = pd.read_csv(file)

    if limit is not None:
        df = df.iloc[:limit]

    # Some cleaning up
    # df_pred = pd.read_csv('zero_shot/study_type_gpt-4o-mini_05-06-05_old.csv')
    # df['prompt'] = df['text'].apply(lambda text: build_prompt(task, text))
    # df['prediction_text'] = df_pred['prediction_text']
    # df['model'] = df_pred['model']

    prompts = []
    predictions = []
    model_specs = []
    if 'llama' in model_name.lower():  # and 'chat' not in model:
        model = LlamaModel(model_name=model_name,
                           use_gpu=True, distributed=False)

    for _, row in df.iterrows():
        prompt = build_class_prompt(task, row['text'])

        if 'Llama-2' in model_name:
            prompt = build_llama_prompt(prompt, system_role_class)

        # Llama chat
        if 'llama' in model_name.lower():  # and 'chat' in model:
            # prompt = build_llama_prompt(prompt, system_role_class)
            model_spec = model.model_name_short
            prediction = model.predict(prompt, temperature=0, do_sample=False)

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
    df_out.to_csv(outfile, index=False, encoding='utf-8')


def make_ner_predictions(model_name: str, outfile: str, limit: int = None):
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
        model = LlamaModel(model_name=model_name,
                           use_gpu=True, distributed=False)

    for _, row in df.iterrows():
        prompt = build_ner_prompt(row['text'])

        if 'Llama-2' in model_name:
            prompt = build_llama_prompt(prompt, system_role_ner)

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


def basic_tokenizer(text: str) -> list[str]:
    text = text.strip()
    doc = nlp(text)
    tokenized = [token.text for token in doc]
    return tokenized


def build_llama_prompt(prompt: str, system_prompt: str) -> str:
    """Build the prompt for the Llama model."""

    # Based in: https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-2/

    # Base model
    # <s>{{ user_prompt }}

    # Meta Llama 2 Chat - single message format
    # <s>[INST] <<SYS>>
    # {{ system_prompt }}
    # <</SYS>>

    # {{ user_message }} [/INST]

    # Meta Llama 2 Chat - multi message format
    # <s>[INST] <<SYS>>
    # {{ system_prompt }}
    # <</SYS>>

    # {{ user_message_1 }} [/INST] {{ model_answer_1 }} </s>
    # <s>[INST] {{ user_message_2 }} [/INST]

    return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n{prompt}[/INST]"


# def parse_ner_prediction(pred: str, tokens: list[str]) -> tuple[str, str]:

#     ner_labels = {
#         'application-area': 'Application Area',
#         'dosage': 'Dosage',
#     }

#     labeled_chuncks = extract_labeled_chunks(pred)

#     token_labels = []
#     aligned_tokens = []
#     token_id = 0

#     for i, (chunk, label) in enumerate(labeled_chuncks):
#         pseudo_tokens = basic_tokenizer(chunk)
#         # Case 1: Chunk is not an NER label
#         if label is None:
#             for ptok in pseudo_tokens:
#                 # Prediction token matches exactly
#                 if ptok == tokens[token_id]:
#                     token_id += 1
#                     token_labels.append('O')
#                     aligned_tokens.append(ptok)
#                 # Prediction token does not match exactly
#                 else:
#                     if tokens[token_id] == '\n':
#                         # If the token is a newline, we can skip it
#                         token_id += 1
#                         token_labels.append('O')
#                         aligned_tokens.append("")

#                     # Case 1.2: Prediction token is a substring of the previous token
#                     # --> consider it as a match but don't add an additinal label
#                     if ptok in tokens[token_id - 1]:
#                         aligned_tokens[-1] += ' ' + ptok
#                     # Case 1.3: Prediction token is a substring of the next token
#                     # --> skip the next token and add the label 'O' for the current token
#                     elif ptok in tokens[token_id + 1]:
#                         # If the ptok is in the next token, we can skip it
#                         token_id += 2
#                         aligned_tokens.append('')
#                         token_labels.append('O')
#                         aligned_tokens.append(ptok)
#                         token_labels.append('O')

#                     elif ptok in tokens[token_id + 2]:
#                         # If the ptok is in the next token, we can skip it
#                         token_id += 3
#                         for i in range(2):
#                             aligned_tokens.append('')
#                             token_labels.append('O')
#                         aligned_tokens.append(ptok)
#                         token_labels.append('O')

#                     # Case 1.1: Prediction token is a substring of the token, e.g. "application" & "application-area"
#                     # --> consider it as a match and continue
#                     elif ptok in tokens[token_id] or tokens[token_id] in ptok:
#                         token_labels.append('O')
#                         token_id += 1
#                         aligned_tokens.append(ptok)
#                     # Case 1.4: Prediction token is a newline
#                     # --> skip the newline and add the label 'O' for the current token

#                     else:
#                         aligned_tokens[-1] += ' ' + ptok

#         # Case 2: Chunk is an NER
#         else:
#             # first = True
#             # for ptok in pseudo_tokens:
#             #     if ptok == tokens[token_id]:
#             #         if first:
#             #             token_labels.append(f'B-{ner_labels[label]}')
#             #             aligned_tokens.append(ptok)
#             #             first = False
#             #         else:
#             #             token_labels.append(f'I-{ner_labels[label]}')
#             #             aligned_tokens.append(ptok)
#             #         token_id += 1
#             #     else:
#             #         if tokens[token_id] == '\n':
#             #             # If the token is a newline, we can skip it
#             #             token_id += 1
#             #             token_labels.append('O')
#             #             aligned_tokens.append("")

#             #         if ptok in tokens[token_id - 1]:
#             #             aligned_tokens[-1] += ' ' + ptok
#             #         elif ptok in tokens[token_id + 1]:
#             #             token_id += 2
#             #             aligned_tokens.append('')
#             #             if first:
#             #                 token_labels.append(f'B-{ner_labels[label]}')
#             #                 first = False
#             #             else:
#             #                 token_labels.append(f'I-{ner_labels[label]}')

#             #             token_labels.append(f'I-{ner_labels[label]}')
#             #             aligned_tokens.append(ptok)

#             #         elif ptok in tokens[token_id] or tokens[token_id] in ptok:
#             #             if first:
#             #                 token_labels.append(f'B-{ner_labels[label]}')
#             #                 aligned_tokens.append(ptok)
#             #                 first = False
#             #             else:
#             #                 token_labels.append(f'I-{ner_labels[label]}')
#             #                 aligned_tokens.append(ptok)
#             #             token_id += 1

#             #         else:
#             #             aligned_tokens[-1] += ' ' + ptok

#             # check if token_labels is the same length as token
#     if len(token_labels) != len(tokens):
#         raise ValueError(
#             f"Token labels length {len(token_labels)} does not match token length {len(tokens)}. Check the prediction: {pred}")

#     return token_labels, aligned_tokens


def parse_ner_predictions(file: str) -> None:
    # Check if file exists
    if not os.path.exists(file):
        raise FileNotFoundError(
            f"The file {file} does not exist. Check the task name.")

    # Check if column 'prediction_text' exists
    df = pd.read_csv(file)
    if 'prediction_text' not in df.columns:
        raise ValueError(
            f"The file {file} does not contain the column 'prediction_text'. Check the file format.")
    # create new folder, with same name as file but without .csv
    name = os.path.splitext(os.path.basename(file))[0]
    output_dir = os.path.join(os.path.dirname(file), name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, row in df.iterrows():
        tokens = ast.literal_eval(row['tokens'])
        if row['id'] == 3638:
            pass
        token_labels, aligned_tokens = parse_ner_prediction(
            row['prediction_text'], tokens)

        aligned_df = pd.DataFrame({
            'tokens': tokens,
            'pred_labels': token_labels,
            'aligned_tokens': aligned_tokens
        })

        # safe with file id.csv in the output directory
        aligned_df.to_csv(os.path.join(
            output_dir, f"{row['id']}.csv"), index=False, encoding='utf-8')

        # save pred_labels
        df.at[i, 'pred_labels'] = str(token_labels)

        # if there is two columns named tokens, remove the second one
        if 'tokens' in df.columns and df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]

        # save the dataframe with the new column
        df.to_csv(file, index=False, encoding='utf-8')


def extract_labeled_chunks(html: str) -> list[tuple[str, str]]:
    # remove ````html` and ` ``` ` from the beginning and end
    html = html.strip().replace('```html', '').replace('```', '')
    if 'h1' in html:
        # strip left from the header tag <h1>
        html = re.sub(r'^.*?<h1>', '<h1>', html, flags=re.DOTALL)

    # replace doupble quotes with single quotes
    html = html.replace('""', '"')
    html = html.strip('"')
    html = html.strip()

    soup = BeautifulSoup(html, "html.parser")
    result = []

    def walk(element, current_label=None):
        for child in element.children:

            if isinstance(child, str):
                result.append((str(child), current_label))
            elif child.name == "span":
                # Get label from class
                label = child.get("class")[0] if child.has_attr(
                    "class") else None
                # Recurse into children with this label
                walk(child, current_label=label)
            else:
                # Recurse into non-span elements
                walk(child, current_label)

    walk(soup)
    return result


def strip_html_tags(text: str) -> str:
    """Strip tags from html text, apart from <span> tags, using regex."""
    text = re.sub(r'<(?!/?span).*?>', '', text)
    return text.strip()


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


def add_tokens():
    test_path = '/home/vera/Documents/Uni/Master/Master_Thesis2.0/PsyNamic-Scale/data/ner_bio/test.csv'
    bioner_path = '/home/vera/Documents/Uni/Master/Master_Thesis2.0/PsyNamic-Scale/zero_shot/ner_gpt-4o-mini_06-06-06.csv'
    df_full = pd.read_csv(test_path)
    df_bioner = pd.read_csv(bioner_path)
    df_bioner = df_bioner.merge(
        df_full[['id', 'tokens', 'ner_tags']], on='id', how='left')
    # save bioner columns and new 'tokens' column
    df_bioner_columns = df_bioner.columns.tolist()
    df_bioner = df_bioner[df_bioner_columns]
    df_bioner.to_csv(bioner_path, index=False, encoding='utf-8')


def main():
    models = [
        "/scratch/vebern/models/Llama-2-13-chat-hf",
        #"/scratch/vebern/models/Llama-2-70-chat-hf",
        "/data/vebern/ma-models/MeLLaMA-13B-chat",
        #"/data/vebern/ma-models/MeLLaMA-70B-chat",
        # "gpt-4o-mini"
    ]
    date = datetime.today().strftime('%d-%m-%d')

    for model_name in models:
        task = "Study Type"
        task_lower = task.lower().replace(' ', '_')
        outfile_class = f"zero_shot/{task_lower}/{task_lower}_{model_name.split('/')[-1]}_{date}.csv"
        # make path
        make_class_predictions(task, model_name, outfile_class)

    for model_name in models:
        outfile_ner = f"zero_shot/ner_{model_name.split('/')[-1]}_{date}.csv"
        make_ner_predictions(model_name, outfile_ner)
        # add_tokens()

    #     tokens = ['Resting', '-', 'state', 'Network', '-', 'specific', 'Breakdown', 'of', 'Functional', 'Connectivity', 'during', 'Ketamine', 'Alteration', 'of', 'Consciousness', 'in', 'Volunteers.^', '\n', 'BACKGROUND', ':', 'Consciousness', '-', 'altering', 'anesthetic', 'agents', 'disturb', 'connectivity', 'between', 'brain', 'regions', 'composing', 'the', 'resting', '-', 'state', 'consciousness', 'networks', '(', 'RSNs', ')', '.', 'The', 'default', 'mode', 'network', '(', 'DMn', ')', ',', 'executive', 'control', 'network', ',', 'salience', 'network', '(', 'SALn', ')', ',', 'auditory', 'network', ',', 'sensorimotor', 'network', '(', 'SMn', ')', ',', 'and', 'visual', 'network', 'sustain', 'mentation', '.', 'Ketamine', 'modifies', 'consciousness', 'differently', 'from', 'other', 'agents', ',', 'producing', 'psychedelic', 'dreaming', 'and', 'no', 'apparent', 'interaction', 'with', 'the', 'environment', '.', 'The', 'authors', 'used', 'functional', 'magnetic', 'resonance', 'imaging', 'to', 'explore', 'ketamine', '-', 'induced', 'changes', 'in', 'RSNs', 'connectivity', '.', 'METHODS', ':', 'Fourteen', 'healthy', 'volunteers', 'received', 'stepwise', 'intravenous', 'infusions', 'of', 'ketamine', 'up', 'to', 'loss', 'of', 'responsiveness', '.', 'Because', 'of', 'agitation', ',', 'data', 'from', 'six', 'subjects', 'were', 'excluded', 'from', 'analysis', '.', 'RSNs', 'connectivity', 'was', 'compared', 'between', 'absence', 'of', 'ketamine', '(', 'wake', 'state', '[', 'W1', ']', ')', ',', 'light', 'ketamine', 'sedation', ',', 'and', 'ketamine', '-', 'induced', 'unresponsiveness', '(', 'deep', 'sedation', '[', 'S2', ']', ')', '.', 'RESULTS', ':', 'Increasing', 'the', 'depth', 'of', 'ketamine', 'sedation', 'from', 'W1', 'to', 'S2', 'altered', 'DMn', 'and', 'SALn', 'connectivity', 'and', 'suppressed', 'the', 'anticorrelated', 'activity', 'between', 'DMn', 'and', 'other', 'brain', 'regions', '.', 'During', 'S2', ',', 'DMn', 'connectivity', ',', 'particularly', 'between', 'the', 'medial', 'prefrontal', 'cortex', 'and', 'the', 'remaining', 'network', '(', 'effect', 'size', 'β', '[', '95', '%', 'CI', ']', ':', 'W1', '=', '0.20', '[', '0.18', 'to', '0.22', ']', ';', 'S2', '=', '0.07', '[', '0.04', 'to', '0.09', ']', ')', ',', 'and', 'DMn', 'anticorrelated', 'activity', '(', 'e.g.', ',', 'right', 'sensory', 'cortex', ':', 'W1', '=', '-0.07', '[', '-0.09', 'to', '-0.04', ']', ';', 'S2', '=', '0.04', '[', '0.01', 'to', '0.06', ']', ')', 'were', 'broken', 'down', '.', 'SALn', 'connectivity', 'was', 'nonuniformly', 'suppressed', '(', 'e.g.', ',', 'left', 'parietal', 'operculum', ':', 'W1', '=', '0.08', '[', '0.06', 'to', '0.09', ']', ';', 'S2', '=', '0.05', '[', '0.02', 'to', '0.07', ']', ')', '.', 'Executive', 'control', 'networks', ',', 'auditory', 'network', ',', 'SMn', ',', 'and', 'visual', 'network', 'were', 'minimally', 'affected', '.', 'CONCLUSIONS', ':', 'Ketamine', 'induces', 'specific', 'changes', 'in', 'connectivity', 'within', 'and', 'between', 'RSNs', '.', 'Breakdown', 'of', 'frontoparietal', 'DMn', 'connectivity', 'and', 'DMn', 'anticorrelation', 'and', 'sensory', 'and', 'SMn', 'connectivity', 'preservation', 'are', 'common', 'to', 'ketamine', 'and', 'propofol', '-', 'induced', 'alterations', 'of', 'consciousness', '.']

    #     pred = '''"```html
    # <h1>Resting-state Network-specific Breakdown of Functional Connectivity during <span class=""application-area"">Ketamine</span> Alteration of Consciousness in Volunteers.^</h1>
    # <p>BACKGROUND: Consciousness-altering anesthetic agents disturb connectivity between brain regions composing the resting-state consciousness networks (RSNs). The default mode network (DMn), executive control network, salience network (SALn), auditory network, sensorimotor network (SMn), and visual network sustain mentation. <span class=""application-area"">Ketamine</span> modifies consciousness differently from other agents, producing psychedelic dreaming and no apparent interaction with the environment. The authors used functional magnetic resonance imaging to explore <span class=""application-area"">ketamine</span>-induced changes in RSNs connectivity. METHODS: Fourteen healthy volunteers received <span class=""dosage"">stepwise intravenous infusions of ketamine up to loss of responsiveness</span>. Because of agitation, data from six subjects were excluded from analysis. RSNs connectivity was compared between absence of <span class=""application-area"">ketamine</span> (wake state [W1]), light <span class=""application-area"">ketamine</span> sedation, and <span class=""application-area"">ketamine</span>-induced unresponsiveness (deep sedation [S2]). RESULTS: Increasing the depth of <span class=""application-area"">ketamine</span> sedation from W1 to S2 altered DMn and SALn connectivity and suppressed the anticorrelated activity between DMn and other brain regions. During S2, DMn connectivity, particularly between the medial prefrontal cortex and the remaining network (effect size β [95% CI]: W1 = 0.20 [0.18 to 0.22]; S2 = 0.07 [0.04 to 0.09]), and DMn anticorrelated activity (e.g., right sensory cortex: W1 = -0.07 [-0.09 to -0.04]; S2 = 0.04 [0.01 to 0.06]) were broken down. SALn connectivity was nonuniformly suppressed (e.g., left parietal operculum: W1 = 0.08 [0.06 to 0.09]; S2 = 0.05 [0.02 to 0.07]). Executive control networks, auditory network, SMn, and visual network were minimally affected. CONCLUSIONS: <span class=""application-area"">Ketamine</span> induces specific changes in connectivity within and between RSNs. Breakdown of frontoparietal DMn connectivity and DMn anticorrelation and sensory and SMn connectivity preservation are common to <span class=""application-area"">ketamine</span> and propofol-induced alterations of consciousness.</p>
    # ```"'''

    #     token_labels, aligned_tokens = parse_ner_prediction(pred, tokens)

    #     # write into csv file
    #     df = pd.DataFrame({
    #         'tokens': tokens,
    #         'pred_labels': token_labels,
    #         'aligned_tokens': aligned_tokens
    #     })
    #     df.to_csv('aligned_test_sample.csv', index=False, encoding='utf-8')
    # parse_ner_predictions('zero_shot/ner_gpt-4o-mini_06-06-06.csv')


if __name__ == "__main__":
    main()
