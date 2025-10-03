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

# color_map = {
#     'gpt-4o-2024-08-06': '#17a583',
#     'gpt-4o-mini': '#48c9b0',
#     'bert-baseline': '#7f7f7f',
#     'tuned': '#ff7f00',
#     'Llama-3.1-8B-Instruct': '#fdbf6f',
#     'Meta-Llama-3-8B-Instruct': '#fb9a99',
#     'Med-LLaMA3-8B': '#e31a1c',
#     'MeLLaMA-70B-chat': '#1f78b4',
#     'MeLLaMA-13B-chat': '#33a02c',
#     'Llama-2-70b-chat-hf': '#a6cee3',
#     'Llama-2-13b-chat-hf': '#b2df8a',
# }

# Based on seaborn colorblind palette
color_map = {
    'gpt-4o-2024-08-06': '#029e73',
    'gpt-4o-mini': '#029e73',
    'bert-baseline': '#949494',
    'tuned': '#de8f05',
    'Llama-3.1-8B-Instruct': '#de8f05',
    'Meta-Llama-3-8B-Instruct': '#d55e00',
    'Med-LLaMA3-8B': '#d55e00',
    'MeLLaMA-70B-chat': '#0173b2',
    'MeLLaMA-13B-chat': '#56b4e9',
    'Llama-2-70b-chat-hf': '#0173b2',
    'Llama-2-13b-chat-hf': '#56b4e9',
}

texture_map = {
    'MeLLaMA-70B-chat': 'hatch',
    'MeLLaMA-13B-chat': 'hatch',
    'Med-LLaMA3-8B': 'hatch',
}


def make_performance_plot(data: dict, save_path: str = None, metrics_col: str = 'metrics') -> None:
    rows = []
    for model, values in data.items():
        # TODO: ugly fix for bert data format
        if 'bert' in model.lower():
            metrics = values['metrics']
        else:
            metrics = values[metrics_col]
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

    num_metrics = len(metrics)
    num_models = len(unique_models)
    fig_width = num_metrics * max(4, num_models * 1.5)
    fig, axes = plt.subplots(1, num_metrics, figsize=(fig_width, 7), sharey=True)
    plt.subplots_adjust(wspace=0.15)

    if num_metrics == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        ax = axes[i]
        df_sub = df_metrics[df_metrics['metric'] == metric].reset_index(drop=True)

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
        ax.set_xticklabels(df_sub["model"], rotation=30, ha="right")

    handles = [plt.Rectangle((0, 0), 1, 1, color=color)
               for model, color in model_colors.items()]
    labels = list(model_colors.keys())
    fig.legend(handles, labels, title="Model", loc="upper left",
               bbox_to_anchor=(0.8, 0.9), title_fontsize="small")

    plt.subplots_adjust(right=0.8, bottom=0.2)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def make_simple_performance_plot(data: pd.DataFrame, save_path: str = None) -> None:
    """
    Plot a simple horizontal barplot comparing model performance.
    Expects a DataFrame with columns: 'model' and 'performance'.
    """
    df = data.copy()
    df = df.sort_values(by='performance', ascending=False).reset_index(drop=True)

    unique_models = df['model'].unique()
    model_colors = {model: color_map.get(model, sns.color_palette("tab10")[i % 10])
                    for i, model in enumerate(unique_models)}

    fig, ax = plt.subplots(figsize=(8, 6))

    barplot = sns.barplot(
        x="performance",
        y="model",
        data=df,
        palette=[model_colors[model] for model in df['model']],
        ax=ax,
        errorbar=None,
        orient="h"
    )

    # Apply hatching to bars for models in texture_map
    for idx, patch in enumerate(ax.patches):
        model = df.loc[idx, "model"]
        if model in texture_map:
            patch.set_hatch('//')

    ax.set(xlim=(0, 1))
    ax.set_xlabel("Performance Score")
    ax.set_ylabel("Model")
    ax.set_title("Model Performance Comparison")

    for idx, row in df.iterrows():
        ax.text(row["performance"] + 0.01, idx, f"{row['performance']:.3f}",
                va="center", ha="left", color="black", fontsize=10)

    # Single legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=model_colors[model], hatch='//' if model in texture_map else '')
               for model in unique_models]
    labels = list(unique_models)
    fig.legend(handles, labels, title="Model", loc="upper left",
               bbox_to_anchor=(0.8, 0.9), title_fontsize="small")

    plt.subplots_adjust(right=0.8, bottom=0.2)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def make_few_shot_performance_plot(data: dict, save_path: str = None, metric: str = "f1-weighted") -> None:
    """
    Plot a grouped barplot for few-shot performance comparison.
    Input data format:
    {
        "model_name": {
            "condition_name": {
                "metrics": {
                    "f1-weighted": [mean, [ci_low, ci_high]],
                    ...
                },
                ...
            },
            ...
        },
        ...
    }
    """
    # Prepare data for plotting
    # Ensure consistent condition order
    condition_order = ["zero_shot", "selected_1shot", "selected_3shot", "selected_5shot"]
    rows = []
    for model, conditions in data.items():
        # Sort conditions by the specified order, fallback to sorted keys
        sorted_conditions = sorted(conditions.keys(), key=lambda x: condition_order.index(x) if x in condition_order else 100 + list(conditions.keys()).index(x))
        for condition in sorted_conditions:
            results = conditions[condition]
            if "metrics" not in results or metric not in results["metrics"]:
                continue
            mean, (ci_low, ci_high) = results["metrics"][metric]
            rows.append({
                'model': model,
                'condition': condition,
                'mean': float(mean),
                'ci_lower': float(ci_low),
                'ci_upper': float(ci_high)
            })
    df = pd.DataFrame(rows)

    # Set up colors for conditions (keep order consistent)
    unique_conditions = [c for c in condition_order if c in df['condition'].unique()]
    # Add any other conditions not in the order list
    unique_conditions += [c for c in df['condition'].unique() if c not in unique_conditions]
    palette = sns.color_palette("tab10", len(unique_conditions))
    condition_colors = dict(zip(unique_conditions, palette))

    # Make chart wider (increased figsize)
    fig, ax = plt.subplots(figsize=(20, 7))  # wider plot

    # Grouped barplot: x=model, hue=condition
    barplot = sns.barplot(
        x="model",
        y="mean",
        hue="condition",
        data=df,
        palette=condition_colors,
        ax=ax,
        errorbar=None,
        dodge=True,
        hue_order=unique_conditions
    )

    ax.set(ylim=(0, 1))
    ax.set_ylabel(metric.replace("-", " ").capitalize())
    ax.set_xlabel("Model")
    ax.set_title(f"Few-Shot Performance Comparison ({metric})")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

    # Add error bars and mean labels, and CI values
    for container, condition in zip(ax.containers, unique_conditions):
        for bar, (_, row) in zip(container, df[df['condition'] == condition].iterrows()):
            yerr_lower = row["mean"] - row["ci_lower"]
            yerr_upper = row["ci_upper"] - row["mean"]
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_height()
            ax.errorbar(x, y, yerr=[[yerr_lower], [yerr_upper]],
                        fmt='none', color='black', capsize=5)
            # Mean value
            ax.text(x, 0.02, f"{row['mean']:.3f}", ha="center",
                    va="bottom", color="black", fontsize=10)
            # CI lower
            ax.text(x, row["ci_lower"] - 0.03,
                    f"{row['ci_lower']:.3f}", ha="center", va="bottom", color="black", fontsize=8)
            # CI upper
            ax.text(x, row["ci_upper"] + 0.01,
                    f"{row['ci_upper']:.3f}", ha="center", va="bottom", color="black", fontsize=8)

    # Single legend for conditions
    handles = [plt.Rectangle((0, 0), 1, 1, color=color)
               for cond, color in condition_colors.items()]
    labels = list(condition_colors.keys())
    ax.legend(handles, labels, title="Condition", loc="upper left",
              bbox_to_anchor=(1.02, 1), title_fontsize="small")

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
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


def make_performance_box_plot(data: pd.DataFrame, title: str, save_path: str = None):
    mean_performance = data.groupby('model')['performance'].mean().sort_values(ascending=False)
    ordered_models = mean_performance.index.tolist()
    
    plt.figure(figsize=(12, 6))
    box = sns.boxplot(
        data=data,
        x="model",
        y="performance",
        showfliers=True,
        order=ordered_models,
        palette=[color_map.get(model, "#cccccc") for model in ordered_models]
    )
    sns.stripplot(
        data=data,
        x="model",
        y="performance",
        color="black",
        size=3,
        alpha=0.6,
        jitter=True,
        order=ordered_models
    )
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.ylabel("Performance")
    plt.xlabel("Model")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def make_performance_spider_plot(data: pd.DataFrame, title: str, save_path: str = None):
    df_pivot = data.pivot(index="model", columns="task", values="performance")
    tasks = df_pivot.columns.tolist()
    num_tasks = len(tasks)

    angles = np.linspace(0, 2 * np.pi, num_tasks, endpoint=False).tolist()
    angles += angles[:1] 
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for model, row in df_pivot.iterrows():
        values = row.tolist()
        values += values[:1] 
        color = color_map.get(model, None)
        linestyle = '--' if model in texture_map else '-'
        ax.plot(angles, values, label=model, color=color, linestyle=linestyle, linewidth=2)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(tasks, fontsize=10)
    ax.set_ylim(0, 1)
    plt.title(title, size=14, weight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    # data = [[1, 0, 1, 0, 1],
    #         [0, 1, 0, 1, 0],
    #         [1, 1, 0, 0, 1]]
    # id2label = {0: 'Label 1', 1: 'Label 2',
    #             2: 'Label 3', 3: 'Label 4', 4: 'Label 5'}
    # plot_label_distribution(data, id2label, "Label Distribution Example")

    # data = [0, 1, 3, 2, 2, 2]
    # id2label = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    # plot_label_distribution(data, id2label, "Label Distribution Example")

    data = {
        'llama2': {
            'zero_shot': [0.5, [0.4, 0.6]],
            '1_shot': [0.7, [0.6, 0.8]],
            '3_shot': [0.75, [0.65, 0.85]],
            '5_shot': [0.8, [0.7, 0.9]],
        },
        'gpt-4o': {
            'zero_shot': [0.6, [0.5, 0.7]],
            '1_shot': [0.72, [0.62, 0.82]],
            '3_shot': [0.78, [0.68, 0.88]],
            '5_shot': [0.82, [0.3, 0.92]],
        }
    }
    make_few_shot_performance_plot(data)