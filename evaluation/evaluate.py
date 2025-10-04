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



# STRIDE-Lab
def bootstrap(metric: callable, y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """Computes bootstrap confidence intervals."""
    print(f"Running bootstrap for {metric.__name__}...")
    score, ci = bootstrap_ci(
        y_true=y_true,
        y_pred=y_pred,
        metric=metric,
        confidence_level=0.95,
        n_resamples=9999,
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
    Evaluate NER predictions against gold annotations, following the same logic
    as metric/extraction.py, but using tuple-based inputs.

    Args:
        list_pred: List of samples, each sample is a list of (entity, type) tuples.
        list_label: Same structure as list_pred, but gold labels.

    Returns:
        dict with accuracy, precision, recall, and f1 for subject (entity only)
        and event (entity+type).
    """

    # Accumulators
    metrics_sample = {
        "subject_tp": [], "subject_fp": [], "subject_fn": [],
        "event_tp": [], "event_fp": [], "event_fn": [],
        "subject_correct_samples": [], "event_correct_samples": []
    }

    for pred, gold in zip(list_pred, list_label):
        # SUBJECT LEVEL = entity string only
        pred_subject = Counter([ent for ent, _ in pred])
        gold_subject = Counter([ent for ent, _ in gold])

        # EVENT LEVEL = (entity, type) pair
        pred_event = Counter(pred)
        gold_event = Counter(gold)

        # --- SUBJECT metrics ---
        tp_subj = sum((pred_subject & gold_subject).values())
        fp_subj = sum((pred_subject - gold_subject).values())
        fn_subj = sum((gold_subject - pred_subject).values())

        subject_correct = 1 if pred_subject == gold_subject else 0

        # --- EVENT metrics ---
        tp_event = sum((pred_event & gold_event).values())
        fp_event = sum((pred_event - gold_event).values())
        fn_event = sum((gold_event - pred_event).values())

        event_correct = 1 if pred_event == gold_event else 0

        # Record per-sample metrics
        metrics_sample["subject_tp"].append(tp_subj)
        metrics_sample["subject_fp"].append(fp_subj)
        metrics_sample["subject_fn"].append(fn_subj)
        metrics_sample["event_tp"].append(tp_event)
        metrics_sample["event_fp"].append(fp_event)
        metrics_sample["event_fn"].append(fn_event)
        metrics_sample["subject_correct_samples"].append(subject_correct)
        metrics_sample["event_correct_samples"].append(event_correct)

    # ---- Aggregate across samples ----
    num_samples = len(list_pred)
    results = {}

    for col in ["subject", "event"]:
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

