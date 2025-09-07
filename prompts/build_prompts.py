# -*- coding: utf-8 -*-
"""
Filename: gpt4_classification_template.py
Description: Template for GPT-4 classification tasks.
Author: Vera Bernhard
"""
# Prompts based on Chen et al. 2025

# How to prompt LLaMa:
# https://huggingface.co/blog/llama2#how-to-prompt-llama-2
# https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-2/

import json
import os
import random
import pandas as pd
import numpy as np
import ast
from typing import Literal

random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(SCRIPT_DIR, '..', 'data')

system_role_class = "You are a helpful medical expert who is helping to classify medical abstracts."
user_prompt_class = '''***TASK***

{TASK_DESCRIPTION}

***INPUT***

The input is the title and abstract text.

***DOCUMENTATION***

There are {NUMBER_OF_TASKS} {TASK} options. The followings are the options and their definitions:

{TASK_OPTIONS}

***OUTPUT***

The output must be exactly one JSON object, and nothing else. Do not include reasoning, the input or multiple JSONs. The JSON should have relevant values for each option: {VALUES}

Put value 1 if the option applies to the research paper, 0 if it does not apply.

Please note again that {IS_MULTIPLE} can be selected for each research paper.

{OUTPUT_EXAMPLE}

INPUT: {TITLE_ABBSTRACT}

OUTPUT: '''


# based on Hu et al. 2024
system_role_ner = "You are a helpful medical expert who is helping to extract named entities from medical abstracts."

user_prompt_ner = '''###Task

Your task is to generate an HTML version of an input text, marking up specific entities. The entities to be identified are: {ENTITIES}. Use HTML <span> tags to highlight these entities. Each <span> should have a class attribute indicating the type of entity. Return only the HTML output. Do not include any additional text, explanations, or formatting before or after the HTML.

###Entity Markup Guide
{ENTITY_MARKUP_GUIDE}

###Entity Definitions
{ENTITY_DEFINITIONS}

###Annotation Guidelines
{ANNOTATION_GUIDELINES}

{EXAMPLES}INPUT: {TITLE_ABSTRACT}

OUTPUT: '''


def build_class_prompt(
    id: int, 
    task: str, 
    title_abstract: str, 
    few_shot: int = 0, 
    few_shot_strategy: Literal['selected', 'random'] = 'selected'):

    file_path = os.path.join(SCRIPT_DIR, 'classification_description.json')

    with open(file_path, 'r', encoding='utf-8') as f:
        task_descriptions = json.load(f)

    if task not in task_descriptions:
        raise ValueError(f"Task '{task}' not found in the task descriptions.")

    if few_shot > 0:
        output = build_class_examples(
            id, task=task, task_options=task_descriptions[task]['Options'], 
            few_shot_nr=few_shot, few_shot_strategy=few_shot_strategy, shots=task_descriptions[task]['Shots'])

    else:
        output = "Example output format:\n" + json.dumps(
            {key: "" for key in task_descriptions[task]['Options'].keys()},
            indent=4
        )

    task_user_prompt = user_prompt_class.format(
        TASK_DESCRIPTION=task_descriptions[task]['Task_description'],
        NUMBER_OF_TASKS=len(task_descriptions[task]['Options']),
        TASK=task,
        TASK_OPTIONS='\n\n'.join(
            [f"{option}: {desc}" for option, desc in task_descriptions[task]['Options'].items()]),
        VALUES=', '.join(task_descriptions[task]['Options'].keys()),
        IS_MULTIPLE='multiple options' if task_descriptions[
            task]['Is_multilabel'] else 'a single option',
        OUTPUT_EXAMPLE=output,
        TITLE_ABBSTRACT=title_abstract
    )

    return task_user_prompt


def build_ner_prompt(id: int, title_abstract: str, few_shot: int = 0, few_shot_strategy: Literal['selected', 'random'] = 'selected'):
    file_path = os.path.join(SCRIPT_DIR, 'ner_description.json')

    with open(file_path, 'r', encoding='utf-8') as f:
        task_descriptions = json.load(f)

    entity_markup_guide = 'Use '
    annotation_guidelines = ''
    definitions = ''

    for entity, det in task_descriptions.items():
        if entity == 'Shots':
            continue
        entity_markup_guide += f'<span class="{entity.lower().replace(" ", "-")}"> to denote {entity}, '

        definitions += f"{entity} is defined as: {det['Definition']}\n\n"

        annotation_guidelines += f'{entity} should be annotated according to the following criteria:\n'
        for crit in det['Criteria']:
            annotation_guidelines += f'* {crit}\n'
        annotation_guidelines += '\n'

    entity_markup_guide = entity_markup_guide[:-2] + '.'
    definitions = definitions[:-2]
    entities = ', '.join([entity for entity, _ in task_descriptions.items()])
    entities.rstrip(', ')
    annotation_guidelines = annotation_guidelines.rstrip('\n')

    shots = task_descriptions['Shots']

    if few_shot > 0:
        examples = build_ner_examples(id, nr=few_shot, few_shot_strategy=few_shot_strategy, shots=shots)
    else:
        examples = ''

    return user_prompt_ner.format(
        ENTITIES=entities,
        ENTITY_MARKUP_GUIDE=entity_markup_guide,
        ENTITY_DEFINITIONS=definitions,
        ANNOTATION_GUIDELINES=annotation_guidelines,
        TITLE_ABSTRACT=title_abstract,
        EXAMPLES=examples
    )


def markup_entities(tokens, text, labels):
    """
    Mark up named entities in text using <span class="...">...</span> tags.
    tokens: list of tokens
    text: raw string text
    labels: BIO labels for each token
    """
    assert len(tokens) == len(
        labels), "Tokens and labels must have the same length"

    result = []
    current_entity = []
    current_type = None

    for token, label in zip(tokens, labels):
        if label.startswith("B-"):
            # close previous entity if open
            if current_entity:
                result.append(
                    f'<span class="{current_type}">' + " ".join(current_entity) + "</span>")
                current_entity = []

            # start new entity
            # e.g. Application area → application-area
            current_type = label[2:].lower().replace(" ", "-")
            current_entity.append(token)

        elif label.startswith("I-") and current_entity:
            # continue entity
            current_entity.append(token)

        else:  # Outside entity
            if current_entity:
                result.append(
                    f'<span class="{current_type}">' + " ".join(current_entity) + "</span>")
                current_entity = []
            result.append(token)

    # join back into text-like string
    marked_text = " ".join(result)

    # fix spacing before punctuation
    marked_text = (
        marked_text.replace(" ,", ",")
        .replace(" .", ".")
        .replace(" :", ":")
        .replace(" ;", ";")
        .replace(" )", ")")
        .replace("( ", "(")
    )

    return marked_text


def build_ner_examples(id: str, nr: int = 3, few_shot_strategy: Literal['selected', 'random'] = 'selected', shots: list[int] = None):
    if few_shot_strategy == 'selected' and shots is None:
        raise ValueError("If few_shot_strategy is 'selected', shots must be provided.")
    output = '###EXAMPLES\n'

    data_file = os.path.join(data_dir, 'ner_bio', 'test.csv')
    df = pd.read_csv(data_file)
    df['id'] = df['id'].astype(int)

    if few_shot_strategy == 'selected':
        # Check if shots are unique
        if len(shots) != len(set(shots)):
            raise ValueError("Shots must be unique.")
        # Check if id is in shots
        if id in shots:
            # remove id from shots
            shots.remove(id)
        ids = shots[:nr]
        
    elif few_shot_strategy == 'random':
        # check if there is any nan in the 'text' column
        ids = df['id'].tolist()
        ids.remove(id)
        # get nr of random ids
        ids = random.sample(ids, nr)

    examples = df[df['id'].isin(ids)]

    for i in ids:
        text = examples[examples['id'] == i]['text'].values[0]
        output += "INPUT: "
        output += text
        output += "\nOUTPUT: "

        tokens = ast.literal_eval(examples[examples['id'] == i]['tokens'].values[0])
        labels = ast.literal_eval(examples[examples['id'] == i]['ner_tags'].values[0])

        output += markup_entities(tokens, text, labels)
        output += '\n\n'

    output += '###FINAL INPUT TO ANNOTATE\n'
    return output


def build_class_examples(id: str, task: str, task_options: dict = None, few_shot_nr: int = 3, few_shot_strategy: Literal['selected', 'random'] = 'selected', shots: list[int] = None):
    if few_shot_strategy == 'selected' and shots is None:
        raise ValueError("If few_shot_strategy is 'selected', shots must be provided.")

    output = '***EXAMPLES***\n'

    task_lower = task.lower().replace(' ', '_')
    data_file = os.path.join(data_dir, task_lower, 'test.csv')
    meta_file = os.path.join(data_dir, task_lower, 'meta.json')
    meta_data = json.load(open(meta_file, 'r'))
    label2int = meta_data['Int_to_label']
    # make sure keys are int
    label2int = {int(k): v for k, v in label2int.items()}
    df = pd.read_csv(data_file)
    # convert all ids to int
    df['id'] = df['id'].astype(int)

    if few_shot_strategy == 'selected':
        # Check if shots are unique
        if len(shots) != len(set(shots)):
            raise ValueError("Shots must be unique.")
        # Check if id is in shots
        if id in shots:
            # remove id from shots
            shots.remove(id)
        ids = shots[:few_shot_nr]

    elif few_shot_strategy == 'random':
        ids = df['id'].tolist()
        # Make sure id of current example is not in the few-shot examples
        ids.remove(id)
        ids = random.sample(ids, few_shot_nr)

    examples = df[df['id'].isin(ids)]

    for i in ids:
        output += "INPUT: "
        output += examples[examples['id'] == i]['text'].values[0]
        output += "\nOUTPUT: "

        labels = examples[examples['id'] == i]['labels'].values[0]
        # Check if it is onehot encoded or just int
        if isinstance(labels, np.int64) or isinstance(labels, int):
            labels_int = int(labels)
            label = label2int[labels_int]
            # set everything in options to 0 apart from the label
            labels_dict = {key: 0 for key in task_options.keys()}
            labels_dict[label] = 1
            output += json.dumps(labels_dict, indent=4)
        else:
            labels_one_hot = json.loads(labels)
            labels = [label2int[i] for i in range(
                len(labels_one_hot)) if labels_one_hot[i] == 1]
            labels_dict = {key: 0 for key in task_options.keys()}
            for label in labels:
                labels_dict[label] = 1
            output += json.dumps(labels_dict, indent=4)

        output += '\n\n'

    output += '***FINAL INPUT TO CLASSIFY***\n'
    return output


def build_llama2_prompt(prompt: str, system_prompt: str) -> str:
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


def build_llama3_prompt(prompt: str, system_prompt: str) -> str:
    pass    
