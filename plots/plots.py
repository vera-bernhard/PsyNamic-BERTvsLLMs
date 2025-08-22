import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from evaluation.evaluate import get_performance_report
from typing import Union
import matplotlib.ticker as mticker
from matplotlib.axes import Axes
import spacy
import numpy as np

BERT_PRED = '/home/vera/Documents/Uni/Master/Master_Thesis2.0/PsyNamic-Scale/bert_baseline/predictions'

sns.set(style="whitegrid")


def make_performance_plot(data: dict):
    rows = []
    for model, values in data.items():
        metrics = values['metrics']
        for metric, (mean, (ci_low, ci_high)) in metrics.items():
            rows.append({
                'model': model,
                'metric': metric,
                'mean': float(mean),
                'ci_lower': float(ci_low),
                'ci_upper': float(ci_high)
            })
    df_metrics = pd.DataFrame(rows)

    unique_models = df_metrics['model'].unique()
    palette = sns.color_palette("tab10", len(unique_models))
    model_colors = dict(zip(unique_models, palette))

    metrics = df_metrics['metric'].unique()

    fig, axes = plt.subplots(1, len(metrics), figsize=(
        6 * len(metrics), 8), sharey=True)

    if len(metrics) == 1:
        axes = [axes]  # ensure iterable if only one metric

    for i, metric in enumerate(metrics):
        ax = axes[i]
        df_sub = df_metrics[df_metrics['metric']
                            == metric].reset_index(drop=True)

        sns.barplot(
            x="model",
            y="mean",
            hue="model",
            data=df_sub,
            palette=model_colors,
            ax=ax,
            errorbar=None,
            legend=False
        )

        ax.set(ylim=(0, 1))

        for idx, row in df_sub.iterrows():
            yerr_lower = row["mean"] - row["ci_lower"]
            yerr_upper = row["ci_upper"] - row["mean"]
            ax.errorbar(idx, row["mean"],
                        yerr=[[yerr_lower], [yerr_upper]],
                        fmt='none', color='black', capsize=5)
            ax.text(idx, 0.02, f"{row['mean']:.3f}", ha="center",
                    va="bottom", color="black", fontsize=10)
            ax.text(idx, row["ci_lower"] - 0.03,
                    f"{row['ci_lower']:.3f}", ha="center", va="bottom", color="black", fontsize=8)
            ax.text(idx, row["ci_upper"] + 0.01,
                    f"{row['ci_upper']:.3f}", ha="center", va="bottom", color="black", fontsize=8)

        ax.set_title(metric.replace('-', ' ').capitalize())
        ax.set_ylabel(metric if i == 0 else "")
        ax.set_xlabel("")
        ax.set_xticks(range(len(df_sub["model"])))
        ax.set_xticklabels(df_sub["model"], rotation=15, ha="right")

    # Single legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=color)
               for model, color in model_colors.items()]
    labels = list(model_colors.keys())
    fig.legend(handles, labels, title="Model", loc="upper left",
               bbox_to_anchor=(0.8, 0.9), title_fontsize="small")

    plt.subplots_adjust(right=0.8, bottom=0.2)
    plt.show()


def plot_label_distribution(
    labels_encoded: Union[list[list[int]], list[int]],
    id2label: dict,
    title: str,
    ax: Axes = None
) -> None:
    """Plot the distribution of labels in a dataset. Labels encoded are either one-hot encoded (list of lists) or single-label encoded (list of integers).
    """
    # Flatten labels for multilabel one-hot, or use directly for single-label id encoding
    if isinstance(labels_encoded[0], list):
        flat_labels = []
        for sublist in labels_encoded:
            flat_labels.extend(
                [i for i, val in enumerate(sublist) if val == 1])
    else:
        flat_labels = labels_encoded

    all_label_ids = sorted(id2label.keys())
    label_counts = pd.Series(flat_labels).value_counts().reindex(
        all_label_ids, fill_value=0)
    label_names = [id2label[i] for i in all_label_ids]

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        x=label_names,
        y=label_counts.values,
        hue=label_names,
        ax=ax
    )
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10, color='black')
        
    ax.set_title(title)
    ax.set_xlabel("Labels")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45)
    # plt.setp(ax.get_xticklabels(), rotation=45)  # <-- Rotate x labels properly


    if fig is not None:
        plt.show()


def plot_length_histogram(data: list[str], title: str, ax: Axes = None, label: str = None, stats_pos: float = 0.95) -> None:
    """Plot a histogram of the lengths of strings in the data, with mean and min/max annotated."""
    nlp = spacy.load("en_core_web_sm")
    lengths = [len(nlp(item)) for item in data]

    mean_length = int(np.mean(lengths))
    min_length = int(np.min(lengths))
    max_length = int(np.max(lengths))

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(lengths, bins=30, kde=True, ax=ax, label=label, alpha=0.5)
    ax.set_xlabel("Length of Strings (tokens)")
    ax.set_ylabel("Frequency")
    if label is None:
        ax.set_title(title)

    # Add a text box with stats for each split, offset vertically
    stats_text = f"{label} stats:\nMean: {mean_length}\nMin: {min_length}\nMax: {max_length}"
    ax.text(0.98, stats_pos, stats_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))

    if fig is not None:
        plt.show()

if __name__ == "__main__":
    data = [[1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
            [1, 1, 0, 0, 1]]
    id2label = {0: 'Label 1', 1: 'Label 2',
                2: 'Label 3', 3: 'Label 4', 4: 'Label 5'}
    plot_label_distribution(data, id2label, "Label Distribution Example")

    data = [0, 1, 3, 2, 2, 2]
    id2label = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    plot_label_distribution(data, id2label, "Label Distribution Example")
