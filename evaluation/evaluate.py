#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: evaluate.py
Description: ...
Author: Vera Bernhard
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from confidenceinterval.bootstrap import bootstrap_ci
from joblib import Parallel, delayed
from collections import Counter
from nervaluate import Evaluator
from collections import defaultdict
import random
from typing import Callable



# STRIDE-Lab
def bootstrap(metric: callable, y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """Computes bootstrap confidence intervals."""
    print(f"Running bootstrap for {metric.__name__}...")
    score, ci = bootstrap_ci(
        y_true=y_true,
        y_pred=y_pred,
        metric=metric,
        confidence_level=0.95,
        n_resamples=1000,
        method="bootstrap_bca",
        random_state=42,
    )
    return score, ci


# STRIDE-Lab
def custom_f1(true_labels: np.ndarray, pred_labels: np.ndarray):
    return f1_score(true_labels, pred_labels, average="weighted", zero_division=0)


# STRIDE-Lab
def custom_accuracy(true_labels: np.ndarray, pred_labels: np.ndarray):
    return accuracy_score(true_labels, pred_labels)


# STRIDE-Lab
def custom_precision(true_labels: np.ndarray, pred_labels: np.ndarray):
    return precision_score(true_labels, pred_labels, average="weighted", zero_division=0)


# STRIDE-Lab
def custom_recall(true_labels: np.ndarray, pred_labels: np.ndarray):
    return recall_score(true_labels, pred_labels, average="weighted", zero_division=0)


# STRIDE-Lab
def get_metrics_ci(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Computes evaluation metrics with confidence intervals in parallel."""
    from joblib import Parallel, delayed

    metric_funcs = [
        ("f1-weighted", custom_f1),
        ("accuracy", custom_accuracy),
        ("precision", custom_precision),
        ("recall", custom_recall),
    ]

    def run_bootstrap(name, func):
        print(f"Computing {name} score and confidence interval...")
        score, ci = bootstrap(func, y_true, y_pred)
        return name, (score, ci)

    results = Parallel(n_jobs=4)(
        delayed(run_bootstrap)(name, func) for name, func in metric_funcs
    )

    metric_dict = dict(results)
    return metric_dict


def get_performance_report(col_tru: str, col_pred: str, df: pd.DataFrame) -> dict:
    """Generates a performance report with metrics and confusion matrix."""

    # Check if columns exist
    if col_tru not in df.columns or col_pred not in df.columns:
        raise ValueError(
            f"Columns '{col_tru}' and '{col_pred}' must be present in the DataFrame. Available columns: {df.columns.tolist()}")

    # Count and remove empty predictions
    nr_empty_tru = df[col_pred].apply(lambda x: pd.isna(x) or x == "").sum()
    df = df[~df[col_pred].isna() & (df[col_pred] != "")]

    # Read labels from string to numpy arrays
    df.loc[:, col_tru] = df[col_tru].apply(
        lambda x: eval(x) if isinstance(x, str) else x)
    df.loc[:, col_pred] = df[col_pred].apply(
        lambda x: eval(x) if isinstance(x, str) else x)
    y_true = np.array(df[col_tru].tolist())
    y_pred = np.array(df[col_pred].tolist())

    metrics = get_metrics_ci(y_true, y_pred)

    # # Confusion matrix
    # conf_matrix = confusion_matrix(y_true, y_pred)

    report = {
        "metrics": metrics,
        # "confusion_matrix": conf_matrix.tolist(),
        # "nr_empty_tru": nr_empty_tru,
    }

    return report



def evaluate_ner_extraction(
    list_pred: list[list[tuple[str, str]]],
    list_label: list[list[tuple[str, str]]]
) -> dict[str, float]:
    """
    Evaluate NER with exact (entity, type) matches.
    
    Returns a flat dictionary with:
    - overall metrics: f1_overall, precision_overall, recall_overall, accuracy_overall
    - per-type metrics: f1_<type>, precision_<type>, recall_<type>, accuracy_<type>
    """
    
    n_samples = len(list_pred)
    
    # Overall counts
    overall_tp = 0
    overall_fp = 0
    overall_fn = 0
    overall_correct_samples = 0  # for accuracy
    
    # Per-type counts
    type_counts = defaultdict(lambda: {"tp":0, "fp":0, "fn":0, "correct_samples":0})
    
    for pred_sample, label_sample in zip(list_pred, list_label):
        pred_set = set(pred_sample)
        label_set = set(label_sample)
        
        # Overall TP/FP/FN
        tp = len(pred_set & label_set)
        fp = len(pred_set - label_set)
        fn = len(label_set - pred_set)
        overall_tp += tp
        overall_fp += fp
        overall_fn += fn
        
        # Overall accuracy: sample is correct if pred_set == label_set
        if pred_set == label_set:
            overall_correct_samples += 1
        
        # Per-type counts
        label_by_type = defaultdict(set)
        pred_by_type = defaultdict(set)
        
        for entity, ent_type in label_set:
            label_by_type[ent_type].add((entity, ent_type))
        for entity, ent_type in pred_set:
            pred_by_type[ent_type].add((entity, ent_type))
        
        all_types = set(label_by_type.keys()).union(pred_by_type.keys())
        for ent_type in all_types:
            pred_entities = pred_by_type.get(ent_type, set())
            label_entities = label_by_type.get(ent_type, set())
            
            tp_type = len(pred_entities & label_entities)
            fp_type = len(pred_entities - label_entities)
            fn_type = len(label_entities - pred_entities)
            
            type_counts[ent_type]["tp"] += tp_type
            type_counts[ent_type]["fp"] += fp_type
            type_counts[ent_type]["fn"] += fn_type
            
            # Accuracy per type: sample is correct for this type if pred_entities == label_entities
            if pred_entities == label_entities:
                type_counts[ent_type]["correct_samples"] += 1
    
    precision_overall = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
    recall_overall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
    f1_overall = 2 * precision_overall * recall_overall / (precision_overall + recall_overall) if (precision_overall + recall_overall) > 0 else 0
    accuracy_overall = overall_correct_samples / n_samples if n_samples > 0 else 0
    
    result = {
        "f1_overall": f1_overall,
        "precision_overall": precision_overall,
        "recall_overall": recall_overall,
        "accuracy_overall": accuracy_overall
    }
    
    for ent_type, counts in type_counts.items():
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        correct_samples = counts["correct_samples"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = correct_samples / n_samples if n_samples > 0 else 0
        
        result[f"f1_{ent_type}"] = f1
        result[f"precision_{ent_type}"] = precision
        result[f"recall_{ent_type}"] = recall
        result[f"accuracy_{ent_type}"] = accuracy

    return result


def evaluate_ner_bio(pred: list[list[str]], true: list[list[str]]) -> dict[str, float]:
    renaming = {
        'I-Application area': 'I-APP',
        'B-Application area': 'B-APP',
        'I-Dosage': 'I-DOS',
        'B-Dosage': 'B-DOS',
        'O': 'O'
    }
    pred = [[renaming.get(label, label) for label in seq] for seq in pred]
    true = [[renaming.get(label, label) for label in seq] for seq in true]

    evaluator = Evaluator(true=true, pred=pred, tags=['APP', 'DOS'], loader='list')
    results = evaluator.evaluate()
    r = {
        'f1 overall - strict': results['overall']['strict'].f1,
        'f1 overall - partial': results['overall']['partial'].f1,
        'precision overall - strict': results['overall']['strict'].precision,
        'precision overall - partial': results['overall']['partial'].precision,
        'recall overall - strict': results['overall']['strict'].recall,
        'recall overall - partial': results['overall']['partial'].recall,
        'f1 APP - strict': results['entities']['APP']['strict'].f1,
        'f1 APP - partial': results['entities']['APP']['partial'].f1,
        'f1 DOS - strict': results['entities']['DOS']['strict'].f1,
        'f1 DOS - partial': results['entities']['DOS']['partial'].f1,
        'precision APP - strict': results['entities']['APP']['strict'].precision,
        'precision APP - partial': results['entities']['APP']['partial'].precision,
        'precision DOS - strict': results['entities']['DOS']['strict'].precision,
        'precision DOS - partial': results['entities']['DOS']['partial'].precision,
        'recall APP - strict': results['entities']['APP']['strict'].recall,
        'recall APP - partial': results['entities']['APP']['partial'].recall,
        'recall DOS - strict': results['entities']['DOS']['strict'].recall,
        'recall DOS - partial': results['entities']['DOS']['partial'].recall,
    }

    return r

def ner_error_analysis(pred: list[list[str]], true: list[list[str]]) -> dict[str, int]:
    renaming = {
        'I-Application area': 'I-APP',
        'B-Application area': 'B-APP',
        'I-Dosage': 'I-DOS',
        'B-Dosage': 'B-DOS',
        'O': 'O'
    }
    pred = [[renaming.get(label, label) for label in seq] for seq in pred]
    true = [[renaming.get(label, label) for label in seq] for seq in true]

    evaluator = Evaluator(true=true, pred=pred, tags=['APP', 'DOS'], loader='list')
    result = evaluator.evaluate()
    r = {
        'correct': result['overall']['strict'].correct,
        'incorrect': result['overall']['strict'].incorrect,
        'partial': result['overall']['strict'].partial,
        'spurious': result['overall']['strict'].spurious,
        'missed': result['overall']['strict'].missed
    }
    return r


def bootstrap_metrics(
    eval_fn: Callable,
    preds: list,
    labels: list,
    n_bootstrap: int = 1000,
    ci: float = 95,
    random_state: int = 42
) -> dict[str, dict[str, float]]:

    rng = random.Random(random_state)
    n = len(preds)
    bootstrap_results = []

    for _ in range(n_bootstrap):
        indices = [rng.randrange(n) for _ in range(n)]
        sample_preds = [preds[i] for i in indices]
        sample_labels = [labels[i] for i in indices]

        res = eval_fn(sample_preds, sample_labels)
        bootstrap_results.append(res)

    all_metrics = {}
    alpha = (100 - ci) / 2
    lower_q = alpha
    upper_q = 100 - alpha

    for key in bootstrap_results[0].keys():
        values = np.array([r[key] for r in bootstrap_results])
        all_metrics[key] = {
            "mean": float(np.mean(values)),
            "lower": float(np.percentile(values, lower_q)),
            "upper": float(np.percentile(values, upper_q))
        }

    return all_metrics