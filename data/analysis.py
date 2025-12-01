#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
from evaluation.plots import plot_label_distribution, plot_length_histogram
import numpy as np
import pandas as pd
from typing import Union
import seaborn as sns
import matplotlib.pyplot as plt
from evaluation.evaluate_zero_shot import TASKS


"""
Filename: analyse_data.py
Description: ...
Author: Vera Bernhard
Date: 27-05-2025
"""


# TODO

# Overall
# - Plot input length distributions of all data (based on 'tokens' column in NER data)


# Classification
# - Number of labels per task
# - Frequency of labels per task and per split
# - Total number of samples per task and per split
# - Plot label distribution per task and per split

# NER
# - Total number of entities per label
# - Total number of entities per label and per split
# - Average number of entities per sample
# - Average number of entities per sample and per split
# - Average length of entities (in tokens)
# - Number of abstracts without entities per split
# - Frequency of each BIO tag (B-*, I-*, O)
# - Plot distribution of entity span lengths
# - Check for malformed BIO sequences (e.g., I-* without preceding B-*)



this_file = os.path.abspath(__file__)
PLOT_DIR = 'analysis_plots'
PLOT_DIR = os.path.join(os.path.dirname(this_file), PLOT_DIR)
os.makedirs(PLOT_DIR, exist_ok=True)



this_file = os.path.abspath(__file__)
PLOT_DIR = 'analysis_plots'
PLOT_DIR = os.path.join(os.path.dirname(this_file), PLOT_DIR)
os.makedirs(PLOT_DIR, exist_ok=True)


def load_metadata(file_path: str) -> dict:
    """Load metadata from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_id2label(task: str):
    task_lower = task.lower().replace(" ", "_")
    meta_file = os.path.join(os.path.dirname(this_file), task_lower, "meta.json")
    metadata = load_metadata(meta_file)
    int2label = metadata.get('Int_to_label', {})
    if isinstance(int2label, dict):
        int2label = {int(k): v for k, v in int2label.items()}
    return int2label


def find_rows_with_2labels(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, encoding='utf-8')
    df['labels'] = df['labels'].apply(lambda x: eval(x))
    return df[df['labels'].apply(lambda x: isinstance(x, list) and x.count(1) == 2)]


def get_labels_from_file(file_path: str) -> Union[list, list[list]]:
    df = pd.read_csv(file_path, encoding='utf-8')
    labels = df['labels'].apply(lambda x: eval(x) if isinstance(x, str) else x).tolist()
    return labels


def get_labels_per_task(task: str) -> tuple[tuple[list, list, list], dict]:
    task_lower = task.lower().replace(" ", "_")
    id2label = load_id2label(task)
    if not id2label:
        raise ValueError(f"No labels found for task: {task}")

    task_dir = os.path.join(os.path.dirname(this_file), task_lower)
    train = get_labels_from_file(os.path.join(task_dir, 'train.csv'))
    test = get_labels_from_file(os.path.join(task_dir, 'test.csv'))
    val = get_labels_from_file(os.path.join(task_dir, 'val.csv'))

    return (train, test, val), id2label


def plot_task_label_distribution(task: str) -> None:
    """Plot label distributions for train, test, and validation splits as subplots."""
    (train, test, val), id2label = get_labels_per_task(task)
    splits = [("Train", train), ("Test", test), ("Validation", val)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for ax, (split_name, labels) in zip(axes, splits):
        plot_label_distribution(labels, id2label, f"{split_name} Label Distribution", ax=ax)
        all_label_ids = sorted(id2label.keys())
        label_names = [id2label[i] for i in all_label_ids]
        ax.set_xticks(range(len(label_names)))
        ax.set_xticklabels(label_names, rotation=45, ha='right')

    plt.suptitle(f"{task} Label Distribution", fontsize=16, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(PLOT_DIR, f"{task.lower().replace(' ', '_')}_label_distribution.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def abstract_length_distribution(task: str):
    """Plot abstract (input text) length distributions per split."""
    task_lower = task.lower().replace(" ", "_")
    task_dir = os.path.join(os.path.dirname(this_file), task_lower)
    fig, ax = plt.subplots(figsize=(10, 6))
    stats_positions = [0.95, 0.80, 0.65]

    for i, split in enumerate(['train', 'test', 'val']):
        file_path = os.path.join(task_dir, f'{split}.csv')
        df = pd.read_csv(file_path, encoding='utf-8')
        df['tokens'] = df['tokens'].apply(lambda x: eval(x) if isinstance(x, str) else x)
        plot_length_histogram(
            df['tokens'].tolist(),
            title=f"{task} Abstract Length Distribution",
            ax=ax,
            label=split.capitalize(),
            stats_pos=stats_positions[i]
        )

    ax.set_title(f"{task} Abstract Length Distribution")
    ax.legend(title="Split")
    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, f"{task_lower}_abstract_length_distribution.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)


def overall_length_distribution(task: str = 'ner_bio'):
    """Plot overall NER token length distribution."""
    all_texts = []
    for split in ['train', 'test', 'val']:
        file_path = os.path.join(os.path.dirname(this_file), task, f'{split}.csv')
        df = pd.read_csv(file_path, encoding='utf-8')
        df['tokens'] = df['tokens'].apply(lambda x: eval(x) if isinstance(x, str) else x)
        all_texts.extend(df['tokens'].tolist())

    # report 90th percentile of lengths
    lengths = [len(tokens) for tokens in all_texts]
    p90 = np.percentile(lengths, 90)
    print(f"Overall NER Titel+Abstract Length 90th Percentile: {p90}")
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_length_histogram(all_texts, title='', ax=ax)
    ax.set_title("Overall Titel+Abstract Length Distribution")
    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, "ner_overall_length_distribution.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)


def overall_nr_class_labels(task: str):
    """Compute and print the average number of labels per sample."""
    (train, test, val), id2label = get_labels_per_task(task)
    all_labels = train + test + val

    if isinstance(all_labels[0], list):
        avg_labels_per_sample = np.mean([sum(labels) for labels in all_labels])
        print(f"{task}: Average number of labels per sample = {avg_labels_per_sample:.2f}")
    else:
        print(f"{task}: Single-label classification (1 label per sample).")


def get_ner_data(split: str) -> list[list[str]]:
    """Load NER data from a CSV file."""
    file_path = os.path.join(os.path.dirname(this_file), 'ner_bio', f'{split}.csv')
    df = pd.read_csv(file_path, encoding='utf-8')
    ner_tags = df['ner_tags'].apply(lambda x: eval(x))
    return ner_tags.tolist()

def get_ner_stats(ner_bio_tags: list[list[str]]) -> dict:
    """
    Compute statistics for BIO-tagged NER data.
    Calculates total entities, average entities per sample,
    average entity length, per-entity-type stats, and counts of samples without entities.
    """
    total_entities = total_dosage = total_application = 0
    total_length = total_length_dosage = total_length_application = 0
    samples_with_entities = samples_with_dosage = samples_with_application = 0
    nr_without_entities = nr_without_dosage = nr_without_application = 0

    for tags in ner_bio_tags:
        entities = dosage = application = 0
        i = 0
        while i < len(tags):
            tag = tags[i]
            if tag.startswith("B-"):
                start = i
                entity_type = tag[2:]
                # Count consecutive I- tags of the same entity
                i += 1
                while i < len(tags) and tags[i] == f"I-{entity_type}":
                    i += 1
                end = i
                length = end - start

                total_length += length
                entities += 1

                if entity_type == "Dosage":
                    dosage += 1
                    total_length_dosage += length
                elif entity_type == "Application area":
                    application += 1
                    total_length_application += length
            else:
                i += 1

        total_entities += entities
        total_dosage += dosage
        total_application += application

        samples_with_entities += int(entities > 0)
        samples_with_dosage += int(dosage > 0)
        samples_with_application += int(application > 0)

        nr_without_entities += int(entities == 0)
        nr_without_dosage += int(dosage == 0)
        nr_without_application += int(application == 0)

    n_samples = len(ner_bio_tags)
    n_entities = total_entities if total_entities > 0 else 1
    n_dosage = total_dosage if total_dosage > 0 else 1
    n_application = total_application if total_application > 0 else 1

    return {
        "total_entities": total_entities,
        "avg_entities_per_sample": total_entities / n_samples if n_samples else 0,
        "avg_entities_dosage": total_dosage / n_samples if n_samples else 0,
        "avg_entities_application": total_application / n_samples if n_samples else 0,
        "avg_entity_length": total_length / n_entities,
        "avg_entity_length_dosage": total_length_dosage / n_dosage,
        "avg_entity_length_application": total_length_application / n_application,
        "samples_with_entities": samples_with_entities,
        "samples_with_dosage": samples_with_dosage,
        "samples_with_application": samples_with_application,
        "nr_without_entities": nr_without_entities,
        "nr_without_dosage": nr_without_dosage,
        "nr_without_application": nr_without_application
    }


def plot_ner_tag_distribution(ner_bio_tags: list[list[str]], title: str, save_name: str) -> None:
    """Plot histogram of number of entities per sample."""
    entity_counts = [sum(1 for tag in tags if tag.startswith('B-')) for tags in ner_bio_tags]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(entity_counts, bins=range(max(entity_counts) + 2), color='skyblue', edgecolor='black', discrete=True, ax=ax)
    ax.set_xlabel('Number of Entities per Sample')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.set_xticks(range(max(entity_counts) + 1))
    ax.grid(axis='y')
    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, save_name)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)


#check duplicates in NER data with the id column
def check_ner_duplicates():
    for split in ['train', 'test', 'val']:
        file_path = os.path.join(os.path.dirname(this_file), 'ner_bio', f'{split}.csv')
        df = pd.read_csv(file_path, encoding='utf-8')
        duplicate_ids = df[df.duplicated(subset=['id'], keep=False)]
        if not duplicate_ids.empty:
            print(f"Duplicate IDs found in {split} set:")
            print(duplicate_ids)
        else:
            print(f"No duplicate IDs in {split} set.")
    # check overall duplicates
    all_ids = []
    for split in ['train', 'test', 'val']:
        file_path = os.path.join(os.path.dirname(this_file), 'ner_bio', f'{split}.csv')
        df = pd.read_csv(file_path, encoding='utf-8')
        all_ids.extend(df['id'].tolist())
    overall_duplicates = pd.Series(all_ids).duplicated(keep=False)
    if overall_duplicates.any():
        print("Duplicate IDs found across splits:")
        print(pd.Series(all_ids)[overall_duplicates])

if __name__ == "__main__":

    # Classification tasks
    for task in TASKS:
        print(f"Analyzing task: {task}")
        plot_task_label_distribution(task)
        overall_nr_class_labels(task)

    # NER data
    sets = ['train', 'test', 'val']
    overall_ner_data = []
    for split in sets:
        ner_bio_data = get_ner_data(split)
        ner_stats = get_ner_stats(ner_bio_data)
        print(f"{split.capitalize()} stats:", ner_stats)
        plot_ner_tag_distribution(
            ner_bio_data,
            f"NER BIO Tag Distribution in {split.capitalize()} Set",
            f"ner_{split}_bio_tag_distribution.png"
        )
        overall_ner_data.extend(ner_bio_data)

    # Overall NER analysis
    plot_ner_tag_distribution(overall_ner_data, "NER BIO Tag Distribution - Overall", "ner_overall_bio_tag_distribution.png")
    print("Overall NER stats:", get_ner_stats(overall_ner_data))
    overall_length_distribution('ner_bio')
    check_ner_duplicates()
