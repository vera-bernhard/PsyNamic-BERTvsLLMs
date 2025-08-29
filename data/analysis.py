#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
from plots.plots import plot_label_distribution, plot_length_histogram
import numpy as np
import pandas as pd
from typing import Union
import seaborn as sns
import matplotlib.pyplot as plt

"""
Filename: analyse_data.py
Description: ...
Author: Vera Bernhard
Date: 27-05-2025
"""

# TODO
# Classification
# - Number of labels per task
# - Frequency of labels per task and per split
# - Total number of samples per task and per split
# - Check if split is stratified (label distribution consistency across splits)
# - (Label entropy per task (to measure imbalance))
# - Average/median input length (tokens or characters) per split
# - Rare class analysis (e.g., labels with <10 examples)
# - Plot input length distributions per split
# - Plot label distribution per task and per split

# NER
# - Total number of entities per label
# - Number of entities per label and per split
# - Average number of entities per sample
# - Average number of entities per sample and per split
# - Average length of entities (in tokens)
# - Number of abstracts without entities
# - Frequency of each BIO tag (B-*, I-*, O)
# - Plot distribution of entity span lengths
# - Rare entity type detection
# - (Label entropy over entity types)
# - Check if split is stratified by entity types
# - Check for malformed BIO sequences (e.g., I-* without preceding B-*)

this_file = os.path.abspath(__file__)


def load_metadata(file_path: str) -> dict:
    """Load metadata from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return metadata


def load_id2label(task: str):
    task_lower = task.lower().replace(" ", "_")
    meta_file = os.path.join(
        os.path.dirname(this_file), task_lower, "meta.json")
    metadata = load_metadata(meta_file)
    int2label = metadata.get('Int_to_label', {})
    if isinstance(int2label, dict):
        int2label = {int(k): v for k, v in int2label.items()}
    return int2label


def find_rows_with_2labels(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, encoding='utf-8')
    # Assuming labels are stored as lists in a column named 'labels'
    df['labels'] = df['labels'].apply(lambda x: eval(x))
    # Filter rows where there is two 1 in the list
    filtered_df = df[df['labels'].apply(
        lambda x: isinstance(x, list) and x.count(1) == 2)]
    return filtered_df


def get_labels_from_file(file_path: str) -> Union[list, list[list]]:
    """Extract labels from a CSV file."""
    df = pd.read_csv(file_path, encoding='utf-8')
    labels = df['labels'].apply(lambda x: eval(
        x) if isinstance(x, str) else x).tolist()
    if isinstance(labels[0], list):
        return labels  # Multilabel
    return labels  # Single-label


def get_labels_per_task(task: str) -> tuple[tuple[list, list, list], dict]:
    """Get the labels for a specific task from the metadata."""
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
        plot_label_distribution(
            labels, id2label, f"{split_name} Label Distribution", ax=ax)
        all_label_ids = sorted(id2label.keys())
        label_names = [id2label[i] for i in all_label_ids]
        # Ensure x-ticks and labels match the bars
        ax.set_xticks(range(len(label_names)))
        ax.set_xticklabels(label_names, rotation=45, ha='right')
    plt.tight_layout()
    plt.suptitle(f"{task} Label Distribution", fontsize=16)
    plt.show()


def abstract_length_distribution(task: str):
    task_lower = task.lower().replace(" ", "_")
    task_dir = os.path.join(os.path.dirname(this_file), task_lower)
    fig, ax = plt.subplots(figsize=(10, 6))
    stats_positions = [0.95, 0.80, 0.65]  # Offset for each split's stats box
    for i, split in enumerate(['train', 'test', 'val']):
        file_path = os.path.join(task_dir, f'{split}.csv')
        df = pd.read_csv(file_path, encoding='utf-8')
        plot_length_histogram(
            df['text'].tolist(),
            title=f"{task} Abstract Length Distribution",
            ax=ax,
            label=split.capitalize(),
            stats_pos=stats_positions[i]
        )
    ax.set_title(f"{task} Abstract Length Distribution")
    ax.legend(title="Split")
    plt.tight_layout()
    plt.show()


def get_ner_data(split: str) -> list[list[str]]:
    """Load NER data from a CSV file."""
    file_path = os.path.join(os.path.dirname(
        this_file), 'ner_bio', f'{split}.csv')
    df = pd.read_csv(file_path, encoding='utf-8')
    ner_tags = df['ner_tags'].apply(
        lambda x: eval(x))
    return ner_tags.tolist()

def get_ner_stats(ner_bio_tags: list[list[str]]) -> dict:
    """
    Compute NER statistics for BIO-tagged data.
    Input: list of lists of BIO tags (e.g., ['B-Application area', 'I-Application area', 'O', ...])
    Returns: dict with statistics.
    """
    total_entities = 0
    total_dosage = 0
    total_application = 0
    samples_with_dosage = 0
    samples_with_application = 0
    samples_with_entities = 0
    nr_without_dosage = 0
    nr_without_application = 0
    nr_without_entities = 0

    for tags in ner_bio_tags:
        entities = 0
        dosage = 0
        application = 0
        prev_tag = 'O'
        for tag in tags:
            if tag.startswith('B-'):
                entities += 1
                if tag == 'B-Dosage':
                    dosage += 1
                elif tag == 'B-Application area':
                    application += 1
            prev_tag = tag
        total_entities += entities
        total_dosage += dosage
        total_application += application

        if dosage > 0:
            samples_with_dosage += 1
        else:
            nr_without_dosage += 1

        if application > 0:
            samples_with_application += 1
        else:
            nr_without_application += 1

        if entities > 0:
            samples_with_entities += 1
        else:
            nr_without_entities += 1

    n_samples = len(ner_bio_tags)
    avg_entities_per_sample = total_entities / n_samples if n_samples else 0
    avg_entities_dosage = total_dosage / n_samples if n_samples else 0
    avg_entities_application = total_application / n_samples if n_samples else 0

    return {
        'total_entities': total_entities,
        'avg_entities_per_sample': avg_entities_per_sample,
        'avg_entities_dosage': avg_entities_dosage,
        'avg_entities_application': avg_entities_application,
        'nr_without_dosage': nr_without_dosage,
        'nr_without_application': nr_without_application,
        'nr_without_entities': nr_without_entities
    }


def plot_ner_tag_distribution(ner_bio_tags: list[list[str]], title: str) -> None:
    """Plot a distribution plot of how many entities per sample there are. It should not distinguish between entity types."""
    entity_counts = [sum(1 for tag in tags if tag.startswith('B-')) for tags in ner_bio_tags]

    plt.figure(figsize=(10, 6))
    sns.histplot(entity_counts, bins=range(max(entity_counts) + 2), color='skyblue', edgecolor='black', discrete=True)
    plt.xlabel('Number of Entities per Sample')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.xticks(range(max(entity_counts) + 1))
    plt.grid(axis='y')
    plt.show()
    
if __name__ == "__main__":
    task = "Substances"
    # (test, train, val), id2label = get_labels_per_task(task)
    # print(f"Task: {task}")
    # print(f"ID to Label Mapping: {id2label}")
    # print(f"Train Labels: {train[:5]}")
    # print(f"Test Labels: {test[:5]}")
    # print(f"Validation Labels: {val[:5]}")

    # plot_task_label_distribution(task)
    # abstract_length_distribution(task)

    # df = find_rows_with_2labels('data/study_type/train.csv')
    # print(df['text'])

    # print(find_rows_with_2labels('data/study_conclusion/train.csv'))
    # print(find_rows_with_2labels('data/study_conclusion/test.csv'))
    # print(find_rows_with_2labels('data/study_conclusion/val.csv'))

    ner_bio_data = get_ner_data('test')
    ner_stats = get_ner_stats(ner_bio_data)
    print(ner_stats)
    plot_ner_tag_distribution(ner_bio_data, "NER BIO Tag Distribution in Test Set")


    

