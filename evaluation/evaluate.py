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
from nervaluate import Evaluator, summary_report_ent, summary_report_overall
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
    Evaluate NER predictions against true annotations, using tuple-based inputs.
    Micro averaged, computes aggregated TP/FP/FN, accuracy defined as = correct predictions/all_predictions (correct == entities match exactly)
    Based on BRIDGE paper
    Args:
        list_pred: List of samples, each sample is a list of (entity, type) tuples.
        list_label: Same structure as list_pred, but true labels.

    Returns:
        dict with overall and type-specific accuracy, precision, recall, and f1
        for both entity (ignoring type) and entity_type (entity+type).
    """

    metrics_sample = {
        "entity_tp": [], "entity_fp": [], "entity_fn": [],
        "entity_type_tp": [], "entity_type_fp": [], "entity_type_fn": [],
        "entity_correct_samples": [], "entity_type_correct_samples": []
    }

    per_type_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for pred, true in zip(list_pred, list_label):
        pred_entity = Counter([ent for ent, _ in pred])
        true_entity = Counter([ent for ent, _ in true])

        pred_entity_type = Counter(pred)
        true_entity_type = Counter(true)

        # Entity-level (ignore type)
        tp_entity = sum((pred_entity & true_entity).values())
        fp_entity = sum((pred_entity - true_entity).values())
        fn_entity = sum((true_entity - pred_entity).values())

        entity_correct = 1 if pred_entity == true_entity else 0

        # Entity+Type level (strict)
        tp_entity_type = sum((pred_entity_type & true_entity_type).values())
        fp_entity_type = sum((pred_entity_type - true_entity_type).values())
        fn_entity_type = sum((true_entity_type - pred_entity_type).values())

        entity_type_correct = 1 if pred_entity_type == true_entity_type else 0

        pred_types = Counter([typ for _, typ in pred])
        true_types = Counter([typ for _, typ in true])

        all_types = set(true_types.keys())
        for t in all_types:
            pred_t = Counter([ent for ent, typ in pred if typ == t])
            true_t = Counter([ent for ent, typ in true if typ == t])

            tp_t = sum((pred_t & true_t).values())
            fp_t = sum((pred_t - true_t).values())
            fn_t = sum((true_t - pred_t).values())

            per_type_counts[t]["tp"] += tp_t
            per_type_counts[t]["fp"] += fp_t
            per_type_counts[t]["fn"] += fn_t

        metrics_sample["entity_tp"].append(tp_entity)
        metrics_sample["entity_fp"].append(fp_entity)
        metrics_sample["entity_fn"].append(fn_entity)
        metrics_sample["entity_type_tp"].append(tp_entity_type)
        metrics_sample["entity_type_fp"].append(fp_entity_type)
        metrics_sample["entity_type_fn"].append(fn_entity_type)
        metrics_sample["entity_correct_samples"].append(entity_correct)
        metrics_sample["entity_type_correct_samples"].append(entity_type_correct)

    num_samples = len(list_pred)
    results = {}

    for col in ["entity", "entity_type"]:
        tp = sum(metrics_sample[f"{col}_tp"])
        fp = sum(metrics_sample[f"{col}_fp"])
        fn = sum(metrics_sample[f"{col}_fn"])
        correct = sum(metrics_sample[f"{col}_correct_samples"])

        accuracy = correct / num_samples if num_samples > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

        results[f"accuracy_{col}"] = accuracy * 100
        results[f"precision_{col}"] = precision * 100
        results[f"recall_{col}"] = recall * 100
        results[f"f1_{col}"] = f1 * 100

    for typ, vals in per_type_counts.items():
        tp, fp, fn = vals["tp"], vals["fp"], vals["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        results[f"precision_{typ}"] = precision * 100
        results[f"recall_{typ}"] = recall * 100
        results[f"f1_{typ}"] = f1 * 100

    return results


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
    results, resultsagg, _, _ = evaluator.evaluate()
    r = {
        'f1 overall - strict': results['strict']['f1'],
        'f1 overall - partial': results['partial']['f1'],
        'f1 APP - strict': resultsagg['APP']['strict']['f1'],
        'f1 APP - partial': resultsagg['APP']['partial']['f1'],
        'f1 DOS - strict': resultsagg['DOS']['strict']['f1'],
        'f1 DOS - partial': resultsagg['DOS']['partial']['f1'],
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