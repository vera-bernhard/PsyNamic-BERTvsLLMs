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
import math
from collections import defaultdict
from scipy.stats import t
import matplotlib.patches as mpatches

BERT_PRED = '/home/vera/Documents/Uni/Master/Master_Thesis2.0/PsyNamic-Scale/bert_baseline/predictions'


# Based on seaborn colorblind palette
MODEL_COLOR_MAP = {
    # 'tuned': '#de8f05',
    'Llama-3.1-8B-Instruct': '#ece133',
    'Meta-Llama-3-8B-Instruct': '#d55e00',
    'Med-LLaMA3-8B': '#d55e00',
    'MeLLaMA-70B-chat': '#0173b2',
    'MeLLaMA-13B-chat': '#56b4e9',
    'Llama-2-70b-chat-hf': '#0173b2',
    'Llama-2-13b-chat-hf': '#56b4e9',
    'gemma-3-27b-it': '#cc78bc',
    'medgemma-27b-text-it': '#cc78bc',
    'gpt-4o-2024-08-06': '#029e73',
    'gpt-4o-mini': '#87f5cb',
    'bert-baseline': '#949494',
    'Llama-3.1-8B-Instruct-IFT': "#ece798",
    'Llama-3.1-8B-Instruct-LST': "#a39a15"
}

METRIC_COLOR_MAP = {
    'F1 BIO - Strict': '#0173b2',
    'F1 BIO - Partial': '#66add9',
    'F1 Strict': "#02548a",
    'Precision BIO - Strict': '#029e73',
    'Precision BIO - Partial': "#5ec4a3",
    'Precision Strict': "#027a59",
    'Recall BIO - Strict': '#d55e00',
    'Recall BIO - Partial': "#f29b64",
    'Recall Strict': "#a44500"
}

TEXTURE_MAP = {
    'MeLLaMA-70B-chat': 'hatch',
    'MeLLaMA-13B-chat': 'hatch',
    'Med-LLaMA3-8B': 'hatch',
    'medgemma-27b-text-it': 'hatch',
}

MODEL_ORDER = [
    'Llama-2-13b-chat-hf',
    'MeLLaMA-13B-chat',
    'Llama-2-70b-chat-hf',
    'MeLLaMA-70B-chat',
    'Meta-Llama-3-8B-Instruct',
    'Med-LLaMA3-8B',
    'Llama-3.1-8B-Instruct',
    'gpt-4o-2024-08-06',
    'gpt-4o-mini',
    'gemma-3-27b-it',
    'medgemma-27b-text-it',
    'Llama-3.1-8B-Instruct-IFT',
    'Llama-3.1-8B-Instruct-LST',
    'bert-baseline',
]

condition_order = ["zero_shot", "selected_1shot",
                   "selected_3shot", "selected_5shot"]

COND_RENAME_MAP_SHORT = {
    'zero_shot': '0',
    'selected_1shot': '1',
    'selected_3shot': '3',
    'selected_5shot': '5',
}

COND_RENAME_MAP = {
    'zero_shot': 'zero-shot',
    'selected_1shot': '1-shot',
    'selected_3shot': '3-shot',
    'selected_5shot': '5-shot',
}
COND_ORDER_SHORT = ['0', '1', '3', '5']
COND_ORDER = ['zero-shot', '1-shot', '3-shot', '5-shot']

METRIC_RENAME_MAP = {
    "f1 overall - strict": "F1 BIO - Strict",
    "precision overall - strict": "Precision BIO - Strict",
    "recall overall - strict": "Recall BIO - Strict",
    "f1 overall - partial": "F1 BIO - Partial",
    "precision overall - partial": "Precision BIO - Partial",
    "recall overall - partial": "Recall BIO - Partial",
    "f1_overall": "F1 Strict",
    "precision_overall": "Precision Strict",
    "recall_overall": "Recall Strict",
}


def make_performance_plot(data: dict, task: str, save_path: str = None, metrics_col: str = 'metrics') -> None:
    if isinstance(data, dict):
        rows = []
        for model, values in data.items():
            # allow bert baselines even if not in model_order
            if model not in MODEL_ORDER and 'bert' not in model.lower():
                continue
            metrics = values[metrics_col]
            for metric, (mean, (ci_low, ci_high)) in metrics.items():
                if 'accuracy' in metric.lower():
                    continue
                rows.append({
                    'model': str(model),
                    'metric': str(metric),
                    'mean': float(mean),
                    'ci_lower': float(ci_low),
                    'ci_upper': float(ci_high)
                })
        df_metrics = pd.DataFrame(rows)
    elif isinstance(data, pd.DataFrame):
        df_metrics = data.copy()
    else:
        raise ValueError("Data must be a dict or a pandas DataFrame")

    # compute model colors (use consistent mapping)
    unique_models = list(df_metrics['model'].unique())
    default_palette = sns.color_palette("tab10")
    model_colors = {}
    for i, m in enumerate(unique_models):
        model_colors[m] = MODEL_COLOR_MAP.get(
            m, default_palette[i % len(default_palette)])
    # special-case bert
    for m in unique_models:
        if 'bert' in m.lower():
            model_colors[m] = MODEL_COLOR_MAP['bert-baseline']

    metrics = df_metrics['metric'].unique()
    num_metrics = len(metrics)
    num_models = len(unique_models)
    fig_width = num_metrics * max(4, num_models * 1.5)
    fig, axes = plt.subplots(
        1, num_metrics, figsize=(fig_width, 7), sharey=True)
    plt.subplots_adjust(wspace=0.15)
    plt.suptitle(
        f"Zero-Shot Comparison for {task}", y=0.98)

    if num_metrics == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        ax = axes[i]
        df_sub = df_metrics[df_metrics['metric'] == metric].copy()

        ordered_models = [
            m for m in MODEL_ORDER if m in df_sub['model'].values]
        remaining = [m for m in df_sub['model'].unique()
                     if m not in ordered_models]
        ordered_models += remaining

        df_sub['model'] = pd.Categorical(
            df_sub['model'], categories=ordered_models, ordered=True)
        df_sub = df_sub.sort_values('model').reset_index(drop=True)

        palette = [model_colors.get(m, default_palette[0])
                   for m in ordered_models]

        sns.barplot(
            x="model",
            y="mean",
            hue="model",
            data=df_sub,
            palette=palette,
            ax=ax,
            errorbar=None,
            order=ordered_models,
            dodge=False,
        )

        ax.set(ylim=(0, 1))

        row_lookup = {row['model']: row for _, row in df_sub.iterrows()}

        for x_pos, mdl in enumerate(ordered_models):
            if mdl not in row_lookup:
                continue
            row = row_lookup[mdl]
            mean = row['mean']
            yerr_lower = mean - row['ci_lower']
            yerr_upper = row['ci_upper'] - mean
            ax.errorbar(x_pos, mean,
                        yerr=[[yerr_lower], [yerr_upper]],
                        fmt='none', color='black', capsize=5)
            ax.text(x_pos, 0.02, f"{row['mean']:.3f}", ha="center",
                    va="bottom", color="black", fontsize=10)
            ax.text(x_pos, row['ci_lower'] - 0.03,
                    f"{row['ci_lower']:.3f}", ha="center", va="bottom", fontsize=8)
            ax.text(x_pos, row['ci_upper'] + 0.01,
                    f"{row['ci_upper']:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_title(metric.replace('-', ' ').capitalize())
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks(range(len(ordered_models)))
        ax.set_xticklabels(ordered_models, rotation=30, ha="right")

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
    df = df.sort_values(
        by='performance', ascending=False).reset_index(drop=True)

    unique_models = df['model'].unique()
    model_colors = {model: MODEL_COLOR_MAP.get(model, sns.color_palette("tab10")[i % 10])
                    for i, model in enumerate(unique_models)}

    fig, ax = plt.subplots(figsize=(8, 6))

    barplot = sns.barplot(
        x="performance",
        y="model",
        data=df,
        palette=[model_colors[model] for model in df['model']],
        ax=ax,
        hue="model",
        errorbar=None,
        orient="h"
    )

    # Apply hatching to bars for models in texture_map
    for idx, patch in enumerate(ax.patches):
        model = df.loc[idx, "model"]
        if model in TEXTURE_MAP:
            patch.set_hatch('//')

    ax.set(xlim=(0, 1))
    ax.set_xlabel("Performance Score")
    ax.set_ylabel("Model")
    ax.set_title("Model Performance Comparison")

    for idx, row in df.iterrows():
        ax.text(row["performance"] + 0.01, idx, f"{row['performance']:.3f}",
                va="center", ha="left", color="black", fontsize=10)

    # Single legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=model_colors[model], hatch='//' if model in TEXTURE_MAP else '')
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


def make_zero_shot_scatter_plot(data: pd.DataFrame, title: str, save_path: str = None) -> None:
    """ x axis tasks, y axis performance, x axis grouped by model with dots with error bars next to each other."""
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # determine task order and model order (respect global model_order first)
    tasks = list(data['task'].unique())
    unique_models = list(data['model'].unique())
    ordered_models = [m for m in MODEL_ORDER if m in unique_models]
    ordered_models += [m for m in unique_models if m not in ordered_models]

    n_tasks = len(tasks)
    n_models = len(ordered_models)

    # horizontal offsets to dodge points within each task
    total_width = 0.6  # total width occupied by all dots at one task
    if n_models > 1:
        offsets = np.linspace(-total_width/2, total_width/2, n_models)
    else:
        offsets = np.array([0.0])

    # map tasks to x positions
    x_positions = np.arange(n_tasks)
    task_to_x = {t: x for t, x in zip(tasks, x_positions)}

    # plot each model's points with its own offset and color
    for i, model in enumerate(ordered_models):
        subset = data[data['model'] == model]
        if subset.empty:
            continue
        xs = [task_to_x[t] + offsets[i] for t in subset['task'].tolist()]
        ys = subset['performance'].values
        ci_l = subset['ci_lower'].values
        ci_u = subset['ci_upper'].values
        color = MODEL_COLOR_MAP.get(model, sns.color_palette("tab10")[i % 10])
        marker = 'o'
        ax.scatter(xs, ys, label=model, color=color,
                   s=100, zorder=3, marker=marker)
        # error bars per-point
        yerr_lower = ys - ci_l
        yerr_upper = ci_u - ys
        ax.errorbar(xs, ys, yerr=[yerr_lower, yerr_upper],
                    fmt='none', color=color, capsize=5, zorder=2)

    # formatting
    ax.set_xticks(x_positions)
    ax.set_xticklabels(tasks, rotation=45, ha="right")
    ax.set_xlim(-0.6, n_tasks - 1 + 0.6)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_ylabel("Performance")
    ax.set_xlabel("Task")

    # single legend underneath
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=min(4, len(ordered_models)), title="Model", title_fontsize="small")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def make_few_shot_performance_plot(data: dict, task: str, save_path: str = None, metric: str = "f1-weighted") -> None:
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
    sns.set_style("darkgrid")

    rows = []
    for model, conditions in data.items():
        # Sort conditions by the specified order, fallback to sorted keys
        sorted_conditions = sorted(conditions.keys(), key=lambda x: condition_order.index(
            x) if x in condition_order else 100 + list(conditions.keys()).index(x))
        for condition in sorted_conditions:
            results = conditions[condition]
            if "metrics" not in results or metric not in results["metrics"]:
                rows.append({
                    'model': model,
                    'condition': condition,
                    'mean': results[metric]['mean'],
                    'ci_lower': results[metric]['lower'],
                    'ci_upper': results[metric]['upper']
                })

            else:
                mean, (ci_low, ci_high) = results["metrics"][metric]
                rows.append({
                    'model': model,
                    'condition': condition,
                    'mean': float(mean),
                    'ci_lower': float(ci_low),
                    'ci_upper': float(ci_high)
                })
    df = pd.DataFrame(rows)
    # Remove any models that don't have 1-shot
    models_with_1shot = df.loc[df['condition'] == 'selected_1shot', 'model'].unique()
    df = df[df['model'].isin(models_with_1shot)| df['model'].str.contains('bert', case=False)]  
    # remove any models which are not in model order or don't have a selected_1shot results, but keep bert
    df = df[df['model'].isin(MODEL_ORDER) | df['model'].str.contains('bert', case=False)]
    bert_model = df[df['model'].str.contains('bert', case=False)]['model'].unique()

    # rename conditions for better display
    df['condition'] = df['condition'].map(COND_RENAME_MAP).fillna(df['condition'])
    # order conditions
    df['condition'] = pd.Categorical(df['condition'], 
                                     categories=COND_ORDER,
                                     ordered=True)

    palette = sns.color_palette("tab10", len(df['condition'].cat.categories))

    # Make sure models are ordered according to model_order
    models = [m for m in MODEL_ORDER if m in df['model'].values] + bert_model.tolist()
    df['model'] = pd.Categorical(df['model'],
                                 categories=models,
                                 ordered=True)
    # Make chart wider (increased figsize)
    fig, ax = plt.subplots(figsize=(28, 7))

    barplot = sns.barplot(
        x="model",
        y="mean",
        hue="condition",
        data=df,
        palette=palette,
        ax=ax,
        errorbar=None,
        dodge=True,
        hue_order=COND_ORDER
    )

    ax.set(ylim=(0, 1))
    ax.set_ylabel(METRIC_RENAME_MAP.get(metric, metric.replace("-", " ").capitalize()))
    ax.set_xlabel("Model")
    ax.set_title(f"In-context Learning Comparison for {task}")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

    # Add error bars and mean labels, and CI values
    for container, condition in zip(ax.containers, COND_ORDER):
        # get all rows
        rows = df[df['condition'] == condition]
        # order them by model to match bar order
        rows = rows.sort_values(by="model")
        for bar, (_, row) in zip(container, rows.iterrows()):
            yerr_lower = row["mean"] - row["ci_lower"]
            yerr_upper = row["ci_upper"] - row["mean"]
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_height()
            offset = 0.02
            ax.errorbar(x, y, yerr=[[yerr_lower], [yerr_upper]],
                        fmt='none', color='black', capsize=5)
            # Mean value
            ax.text(x, 0.02, f"{row['mean']:.2f}", ha="center",
                    va="bottom", color="black", fontsize=8)

            # CI lower label: slightly below the lower CI bound
            ax.text(x, row["ci_lower"] - 2 * offset, f"{row['ci_lower']:.2f}",
                    ha="center", va="bottom", fontsize=8,)

            # CI upper label: slightly above the upper CI bound
            ax.text(x, row["ci_upper"] + offset, f"{row['ci_upper']:.2f}",
                    ha="center", va="bottom", fontsize=8)

    # Single legend for conditions
    condition_colors = dict(zip(df['condition'].cat.categories, palette))
    handles = [plt.Rectangle((0, 0), 1, 1, color=color)
               for color in condition_colors.values()]
    labels = list(condition_colors.keys())
    ax.legend(handles, labels, title="Condition", loc="upper left",
              bbox_to_anchor=(1.02, 1), title_fontsize="small")

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def make_few_shot_box_plot(data: pd.DataFrame, task: str, save_path: str = None, metric: str = "f1-weighted") -> None:
    """ Make a grouped box plot for few-shot performance comparison."""
    plt.figure(figsize=(12, 6))

    palette = sns.color_palette("tab10", len(condition_order))
    sns.boxplot(data=data, x="model", y=metric,
                hue="condition", palette=palette)
    plt.title(f"Few-Shot Performance Comparison for {task}")
    plt.ylabel(metric.replace("-", " ").capitalize())
    plt.xlabel("Model")
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_label_distribution(
    labels_encoded: Union[list[list[int]], list[int]],
    id2label: dict,
    title: str,
    ax: plt.Axes = None,
    save_path: str = None
) -> None:
    """Plot the distribution of labels in a dataset. Labels encoded are either one-hot encoded (list of lists) or single-label encoded (list of integers)."""

    if isinstance(labels_encoded[0], list):
        flat_labels = []
        for sublist in labels_encoded:
            flat_labels.extend([i for i, val in enumerate(sublist) if val == 1])
    else:
        flat_labels = labels_encoded

    all_label_ids = sorted(id2label.keys())
    label_counts = pd.Series(flat_labels).value_counts().reindex(all_label_ids, fill_value=0)
    label_names = [id2label[i] for i in all_label_ids]

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        x=label_names,
        y=label_counts.values,
        palette=sns.color_palette("colorblind", n_colors=len(label_names)),
        ax=ax,
        hue=label_names,
        legend=False
    )

    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    for p in ax.patches:
        x = p.get_x() + p.get_width() / 2.0
        y = p.get_y() + p.get_height() / 2.0
        ax.annotate(f'{int(p.get_height())}',
                    (x, y),
                    ha='center', va='center', fontsize=10, color='black', clip_on=False)

    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.tick_params(axis='x', rotation=45)

    if fig is not None:
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def plot_length_histogram(data: list[str], title: str, ax: Axes = None, label: str = None, stats_pos: float = 0.95) -> None:
    """Plot a histogram of the lengths of strings in the data, with mean and min/max annotated."""
    lengths = [len(item) for item in data]

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
    if label:
        stats_text = f"{label} stats:\nMean: {mean_length}\nMin: {min_length}\nMax: {max_length}"
    else:
        stats_text = f"Mean: {mean_length}\nMin: {min_length}\nMax: {max_length}"
    ax.text(0.98, stats_pos, stats_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))

    if fig is not None:
        plt.show()


def make_performance_box_plot(data: pd.DataFrame, title: str, save_path: str = None):
    # Ensure we only keep models present in the global model_order and preserve that order
    df = data.copy()
    df = df[df['model'].isin(MODEL_ORDER)]
    ordered_models = [m for m in MODEL_ORDER if m in df['model'].unique()]

    # Make 'model' categorical with the desired order so plotting respects it
    df['model'] = pd.Categorical(
        df['model'], categories=ordered_models, ordered=True)

    plt.figure(figsize=(6, 6))
    box = sns.boxplot(
        data=df,
        x="model",
        y="performance",
        showfliers=True,
        order=ordered_models,
        palette=[MODEL_COLOR_MAP.get(model, "#cccccc")
                 for model in ordered_models]
    )
    sns.stripplot(
        data=df,
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
    plt.ylabel("F1-Weighted")
    plt.xlabel("Model")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def make_performance_spider_plot(data: pd.DataFrame, title: str, save_path: str = None):
    data = data[data['model'].isin(MODEL_ORDER)]
    df_pivot = data.pivot(index="model", columns="task", values="performance")
    tasks = df_pivot.columns.tolist()
    num_tasks = len(tasks)

    angles = np.linspace(0, 2 * np.pi, num_tasks, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # order models according to model_order
    models_in_df = list(df_pivot.index)
    ordered_models = [m for m in MODEL_ORDER if m in models_in_df]
    ordered_models += [m for m in models_in_df if m not in ordered_models]

    for model, row in df_pivot.iterrows():
        values = row.tolist()
        values += values[:1]
        color = MODEL_COLOR_MAP.get(model, sns.color_palette(
            "tab10")[ordered_models.index(model) % 10])
        linestyle = '--' if model in TEXTURE_MAP else '-'
        ax.plot(angles, values, label=model, color=color,
                linestyle=linestyle, linewidth=2)
        ax.fill(angles, values, alpha=0.1, color=color)

    for i, task in enumerate(tasks):
        task_values = df_pivot[task]
        max_value = task_values.max()

        angle = angles[i]

        offset = 0.05
        ax.text(
            angle,
            max_value + offset,
            f"{max_value:.2f}",       # Zahl mit 2 Nachkommastellen
            fontsize=9,
            ha='center',
            va='bottom',
        )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(tasks, fontsize=10)
    ax.set_ylim(0, 1)
    plt.title(title)
    # place legend to the right outside the plot and make room
    fig.legend(loc='center left',
               bbox_to_anchor=(1.02, 0.5),
               fontsize='small',
               title='Models',)
    plt.subplots_adjust(right=0.78)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def make_few_shot_trend_plot(data: pd.DataFrame, title: str, save_path: str = None, metric: str = "f1-weighted", sharey: bool = True):
    """ - y = tasks
        - x = condition
        - x groupe by model

        multiplot models * tasks with lines with points for each condition
    """
    sns.set_style("darkgrid")

    # Rename
    data['model'] = data['model'].apply(
        lambda x: 'bert-baseline' if 'bert' in x.lower() else x)
    data['condition'] = data['condition'].replace(COND_RENAME_MAP_SHORT)

    # Filter
    skip_models = ['tuned', 'bert-baseline']
    data = data[~data['model'].isin(skip_models)]
    data = data[data['metric'] == metric]
    data = data.rename(columns={'mean': metric})

    # Sorting
    tasks = sorted(data['task'].unique().tolist())
    models = data['model'].unique().tolist()
    models = [m for m in MODEL_ORDER if m in models]
    data['model'] = pd.Categorical(
        data['model'], categories=models, ordered=True)
    data['condition'] = pd.Categorical(
        data['condition'], categories=COND_ORDER_SHORT, ordered=True)

    A4_width, A4_height = 8.27, 11.69
    rows = len(tasks)
    ncols = len(models)

    if sharey:
        g = sns.FacetGrid(
            data,
            col="model",
            row="task",
            sharey=True,
            sharex=True,
            # height=A4_height / nrows,
            # aspect=(A4_width / ncols) / (A4_height / nrows)
        )
        g.map_dataframe(sns.lineplot, x="condition", y=metric, marker="o")

    else:
        g = sns.FacetGrid(
            data,
            col="model",
            row="task",
            sharey=False,
            sharex=True,
            # height=A4_height / nrows,
            # aspect=(A4_width / ncols) / (A4_height / nrows)
        )
        g.map_dataframe(sns.lineplot, x="condition", y=metric, marker="o")

    # fixed_span = 0.25 # desired total y-axis range

    # for ax in g.axes.flat:
    #     # Get current data limits
    #     ymin, ymax = ax.get_ylim()
    #     center = (ymin + ymax) / 2

    #     # Compute new symmetrical limits
    #     new_ymin = center - fixed_span / 2
    #     new_ymax = center + fixed_span / 2

    #     # Apply the fixed-size window
    #     ax.set_ylim(new_ymin, new_ymax)

    if not sharey:
        fixed_tick_interval = 0.05
        for ax in g.axes.flat:
            # Keep y-grid visible and evenly spaced
            ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.7)
            ax.yaxis.set_major_locator(
                mticker.MultipleLocator(fixed_tick_interval))

    # Remove numbers and spans
    for ax in g.axes.flat:
        for spine in ["left", "right", "top", "bottom"]:
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)

    axes = np.array(g.axes)

    # Place model names as xlabels on the bottom row axes, rotated 90 degrees
    for col_idx, model_name in enumerate(models):
        bottom_ax = axes[-1, col_idx]
        bottom_ax.set_xlabel(model_name)

        label = bottom_ax.xaxis.get_label()
        label.set_rotation(15)
        label.set_horizontalalignment('right')
        label.set_verticalalignment('top')
        label.set_rotation_mode('anchor')

    # Place task names to the left of each row (as ylabels), horizontal and centered vertically
    # increase labelpad to avoid clipping
    max_task_len = max((len(str(t)) for t in tasks), default=0)
    labelpad_left = 35 + max(0, (max_task_len - 10) * 3)
    for row_idx, task_name in enumerate(tasks):
        left_ax = axes[row_idx, 0]
        left_ax.set_ylabel(task_name, rotation=0,
                           labelpad=labelpad_left, va='center')

    # Remove any subtitles/titles inside each subplot and the legend
    for ax in axes.flatten():
        ax.set_title("")
        for txt in list(ax.texts):
            ax.texts.remove(txt)

    # Add overall title and adjust layout so title doesn't overlap subplots
    if title:
        g.fig.suptitle(title, y=0.98)
        g.fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        g.fig.tight_layout()

    if save_path:
        g.fig.savefig(save_path, bbox_inches='tight')
        plt.close(g.fig)
    else:
        plt.show()


def make_few_shot_delta_plot(data: pd.DataFrame, title: str, save_path: str = None, metric: str = "f1-weighted"):
    """ y axis: performance improvement over zero-shot, x axis: condition, grouped by model and task, grid of models * tasks"""
    sns.set_style("darkgrid")
    # Rename
    data['model'] = data['model'].apply(
        lambda x: 'bert-baseline' if 'bert' in x.lower() else x)
    data['condition'] = data['condition'].replace(COND_RENAME_MAP_SHORT)

    # Filter
    skip_models = ['tuned', 'bert-baseline']
    data = data[~data['model'].isin(skip_models)]
    data = data[data['metric'] == metric].copy()
    data = data.rename(columns={'mean': metric})
    # Remove any models that don't have any 1 shot results
    models_with_1shot = data[data['condition']
                             == '1']['model'].unique().tolist()
    data = data[data['model'].isin(models_with_1shot)]

    # Sorting
    tasks = sorted(data['task'].unique().tolist())
    data['task'] = pd.Categorical(data['task'], categories=tasks, ordered=True)
    models = data['model'].unique().tolist()
    models = [m for m in MODEL_ORDER if m in models]
    data = data[data['model'].isin(models)]

    # Calculate relative improvements compared to zero-shot
    for model in models:
        for task in tasks:
            baseline = data[
                (data['model'] == model)
                & (data['task'] == task)
                & (data['condition'] == '0')
            ]

            for cond in COND_ORDER_SHORT[1:]:
                current = data[
                    (data['model'] == model)
                    & (data['task'] == task)
                    & (data['condition'] == cond)
                ]
                if not baseline.empty and not current.empty:
                    improvement = current[metric].values[0] - \
                        baseline[metric].values[0]
                    data.loc[current.index, metric] = improvement

    

    # remove all zero-shot rows as they are now baselines
    data = data[data['condition'] != '0']
    # add color column based on positive/negative improvement
    data['color'] = np.where(data[metric] > 0, 'green', 'red')

    data['model'] = pd.Categorical(
        data['model'], categories=models, ordered=True)
    data['condition'] = pd.Categorical(
        data['condition'], categories=COND_ORDER_SHORT[1:], ordered=True)

    A4_width, A4_height = 8.27, 11.69
    nrows = len(tasks)
    ncols = len(models)

    g = sns.FacetGrid(
        data,
        col="model",
        row="task",
        sharey=True,
        sharex=True,
        aspect=0.5 if len(tasks) == 1 else 0.7,
        # height=A4_height / nrows,
        # aspect=(A4_width / ncols) / (A4_height / nrows)
    )
    g.map_dataframe(sns.barplot, x="condition", y=metric, hue="color",
                    dodge=False, palette={"green": "green", "red": "red"})

    for ax in g.axes.flat:
        for patch in ax.patches:
            height = patch.get_height()
            if height == 0:
                continue
            x = patch.get_x() + patch.get_width() / 2
            y = height + 0.01 if height >= 0 else height - 0.01
            va = 'bottom' if height >= 0 else 'top'
            ax.text(x, y, f'{height:.2f}', ha='center', va=va, fontsize=7, rotation=90)

    # Remove numbers and spans
    for row_idx, row_axes in enumerate(g.axes):
        for col_idx, ax in enumerate(row_axes):
            for spine in ["right", "top"]:
                ax.spines[spine].set_visible(False)

            if col_idx == 0:
                ax.tick_params(axis="y", which="both",
                               left=True, labelleft=True)
                ax.spines["left"].set_visible(True)
                ax.set_ylabel("")
            else:
                ax.tick_params(axis="y", which="both",
                               left=False, labelleft=False)
                ax.spines["left"].set_visible(False)

    axes = np.array(g.axes)

    # Model names at the bottom, no other xlabels
    for col_idx, model_name in enumerate(models):
        bottom_ax = axes[-1, col_idx]
        bottom_ax.set_xlabel(model_name)

        label = bottom_ax.xaxis.get_label()
        label.set_rotation(15)
        label.set_horizontalalignment('right')
        label.set_verticalalignment('top')
        label.set_rotation_mode('anchor')

    # Task names to the left
    if len(tasks) > 1:
        max_task_len = max((len(str(t)) for t in tasks), default=0)
        labelpad_left = 35 + max(0, (max_task_len - 10) * 3)
        for row_idx, task_name in enumerate(tasks):
            left_ax = axes[row_idx, 0]
            left_ax.set_ylabel(task_name, rotation=0,
                               labelpad=labelpad_left, va='center')
            for cond in range(1, len(models)):
                axes[row_idx, cond].set_ylabel("Δ F1")
    g.set_titles("")
    # Add overall title if provided and adjust layout
    if title:
        if len(tasks) == 1:
            g.fig.suptitle(title, y=0.85)
        else: 
            g.fig.suptitle(title, y=0.95)
        g.fig.tight_layout(rect=[0, 0, 1, 0.95])
        if len(tasks) == 1:# single row, adjust left to avoid clipping
            g.fig.subplots_adjust(left=0.35)
    else:
        g.fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        g.fig.savefig(save_path, bbox_inches='tight')
        plt.close(g.fig)
    else:
        plt.show()


def make_few_shot_delta_task_averaged_plot(data, title, save_path=None, metric="f1-weighted"): 
    data = data.copy()
    data['model'] = data['model'].apply(lambda x: 'bert-baseline' if 'bert' in x.lower() else x)
    data['condition'] = data['condition'].replace(COND_RENAME_MAP_SHORT)
    # Filter metric and remove irrelevant models
    skip_models = ['tuned', 'bert-baseline']
    data = data[~data['model'].isin(skip_models)]
    data = data[data['metric'] == metric].copy()
    data = data.rename(columns={'mean': metric})

    # Remove models without 1-shot results
    # remove medllama
    # data = data[~data['model'].str.contains('med-llama', case=False)]
    models_with_1shot = data[data['condition'] == '1']['model'].unique().tolist()
    data = data[data['model'].isin(models_with_1shot)]

    # Sorting
    tasks = sorted(data['task'].unique().tolist())
    models = data['model'].unique().tolist()
    models = [m for m in MODEL_ORDER if m in models]
    data = data[data['model'].isin(models)]

    # Calculate relative improvements vs zero-shot
    for model in models:
        for task in tasks:
            baseline = data[(data['model']==model) & (data['task']==task) & (data['condition']=='0')]
            for cond in COND_ORDER_SHORT[1:]:
                current = data[(data['model']==model) & (data['task']==task) & (data['condition']==cond)]
                if not baseline.empty and not current.empty:
                    improvement = current[metric].values[0] - baseline[metric].values[0]
                    data.loc[current.index, metric] = improvement

    # Remove zero-shot rows
    data = data[data['condition'] != '0']


    agg = (
        data.groupby(["model", "condition"])
            .agg(
                mean_delta=(metric, "mean"),
                std_delta=(metric, "std"),
                n_tasks=(metric, "count")
            )
            .reset_index()
    )

    # t-based 95% CI
    agg['t_val'] = agg['n_tasks'].apply(lambda n: t.ppf(0.975, df=n-1))
    agg['ci95'] = agg['t_val'] * agg['std_delta'] / np.sqrt(agg['n_tasks'])
    agg["color"] = np.where(agg["mean_delta"] > 0, "green", "red")
    agg["model"] = pd.Categorical(agg["model"], categories=models, ordered=True)
    agg["condition"] = pd.Categorical(agg["condition"], categories=COND_ORDER_SHORT[1:], ordered=True)

    sns.set_style("darkgrid")
    n_models = len(models)
    nr_cols = 6
    nr_rows = 2
    fig, axes = plt.subplots(nr_rows, nr_cols, sharey=True, figsize=(12, 4*nr_rows), constrained_layout=True)
    if n_models == 1:
        axes = [axes]  # make it iterable
    axes = axes.flatten()
    for ax, model in zip(axes, models):
        sub = agg[agg["model"] == model]
        sub = sub.set_index("condition").reindex(COND_ORDER_SHORT[1:]).reset_index()
        xs = np.arange(len(sub))
        heights = sub["mean_delta"].fillna(0).values
        errors = sub["ci95"].fillna(0).values
        colors = sub["color"].fillna("grey").values
        ax.bar(xs, heights, color=colors)

        # Only draw error bars where values exist
        valid = ~sub["mean_delta"].isna().values
        ax.errorbar(xs[valid], heights[valid], yerr=errors[valid],
                    color='black', fmt='none', capsize=2)

        # X-axis labels
        ax.set_xticks(xs)
        ax.set_xticklabels(sub["condition"].values, ha='right')
        ax.set_xlabel(model, rotation=30, ha='right', va='top')
        offset = 0.01
        # for x, y in zip(xs, heights):
        #     # place label relative to x-axis
        #     y_text = y + offset if y >= 0 else y - offset
        #     va = 'bottom' if y >= 0 else 'top'
        #     ax.text(x, y_text, f"{y:.2f}", ha='center', va=va, fontsize=8)
        #     # add text to error bars
        #     ax.text(x, y_text + (errors[x] + offset if y >= 0 else -errors[x] - offset), f"±{errors[x]:.2f}", ha='center', va=va, fontsize=8)

        # Labels
        ax.set_xlabel(model)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
        # sub_raw = data[(data["model"] == model)]
        # for i, cond in enumerate(COND_ORDER_SHORT[1:]):
        #     vals = sub_raw[sub_raw["condition"] == cond][metric].values

        #     if len(vals) == 0:
        #         continue

        #     # Jitter so points don’t overlap perfectly
        #     jitter = np.random.uniform(-0.04, 0.04, size=len(vals))

        #     # Color by sign
        #     colors_pts = ["green" if v >= 0 else "red" for v in vals]

        #     ax.scatter(
        #         np.full_like(vals, i) + jitter,  # x positions
        #         vals,                            # y values
        #         s=6,                            # dot size
        #         color=colors_pts,
        #         alpha=0.5,
        #         zorder=5
        #     )
    axes[0].set_ylabel(f"Δ {metric}")
    # set y limits,0.1 and -0-3
    axes[0].set_ylim(-0.2, 0.2)
    # remove empty subplots
    for i in range(n_models, len(axes)):
        fig.delaxes(axes[i])

    fig.suptitle(title)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def make_few_shot_avg_plot(data: pd.DataFrame, title: str, save_path: str = None, metric: str = "f1-weighted"):
    """ x axis: model, y axis: average performance across tasks, grouped by condition with error bars"""
    # Filter and rename
    data = data[data['metric'] == metric]
    data = data.rename(columns={'mean': metric})
    data['model'] = data['model'].apply(
        lambda x: 'bert-baseline' if 'bert' in x.lower() else x)
    skip_models = ['tuned', 'bert-baseline']
    data = data[~data['model'].isin(skip_models)]
    data['condition'] = data['condition'].replace(COND_RENAME_MAP)

    # average across tasks
    df_avg = (
        data.groupby(['model', 'condition'])[metric]
        .agg(['mean', 'std'])
        .reset_index()
    )

    # order models and conditions
    df_avg = df_avg[df_avg['model'].isin(MODEL_ORDER)]
    df_avg['model'] = pd.Categorical(
        df_avg['model'], categories=MODEL_ORDER, ordered=True)

    present_conditions = list(df_avg['condition'].unique())
    hue_order = [c for c in condition_order if c in present_conditions] + [
        c for c in present_conditions if c not in condition_order
    ]

    df_avg['condition'] = pd.Categorical(
        df_avg['condition'], categories=hue_order, ordered=True)
    df_avg = df_avg.sort_values(['model', 'condition']).reset_index(drop=True)
    # make bar plot with grouped bars for each model and condition
    plt.figure(figsize=(12, 6))
    cond_color = sns.color_palette("tab10", len(hue_order))
    # map each condition to a color so we can index by condition name (hue labels are strings)
    cond_color_map = dict(zip(hue_order, cond_color))
    ax = sns.barplot(
        data=df_avg,
        x='model',
        y='mean',
        hue='condition',
        palette=cond_color_map,
        dodge=True,
        hue_order=hue_order,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

    bars = ax.patches
    for bar, (_, row) in zip(bars, df_avg.iterrows()):
        x = bar.get_x() + bar.get_width() / 2
        # subtsract stddev for lower error bar, add stddev for upper error bar
        y = bar.get_height()
        ax.errorbar(x, y, yerr=[[row['std']], [row['std']]],
                    fmt='none', color='black', capsize=5)

    # Single legend for conditions
    handles = [plt.Rectangle((0, 0), 1, 1, color=cond_color_map[c])
               for c in hue_order]
    labels = list(hue_order)
    ax.legend(handles, labels, title="Condition", loc="upper left",
              bbox_to_anchor=(1.02, 1), title_fontsize="small")

    # y range 0-1
    ax.set(ylim=(0, 1))
    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel("F1 Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def make_few_shot_parallel_plot(data: pd.DataFrame, title: str, save_path: str = None, metric: str = "f1-weighted"):
    """x-axis: condition, y-axis: performance, lines: models, grid with one subplot per task."""
    
    sns.set_style("darkgrid")

    # Filter and rename
    data = data[data['metric'] == metric]
    data = data.rename(columns={'mean': metric})
    data['model'] = data['model'].apply(
        lambda x: 'bert-baseline' if 'bert' in x.lower() else x)
    data['condition'] = data['condition'].replace(COND_RENAME_MAP_SHORT)


    # Extract bert-baseline performance per task
    bert_metric_per_task = data[data['model'] == 'bert-baseline'][['task', metric]]

    # Skip unwanted models in the main lines
    skip_models = ['tuned', 'bert-baseline']
    data = data[~data['model'].isin(skip_models)]


    # Filter and order models
    models = [m for m in MODEL_ORDER if m in data['model'].unique()]
    data['model'] = pd.Categorical(
        data['model'], categories=models, ordered=True)
    data['condition'] = pd.Categorical(
        data['condition'], categories=COND_ORDER_SHORT, ordered=True)

    tasks = sorted(data['task'].unique())
    n_tasks = len(tasks)
    # Layout: roughly square grid
    ncols = min(4, n_tasks)
    nrows = math.ceil(n_tasks / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(4 * ncols, 3 * nrows), sharey=True)
    axes = np.array(axes).flatten()

    for i, task in enumerate(tasks):
        ax = axes[i]
        subset = data[data['task'] == task]
        sns.lineplot(
            data=subset,
            x='condition',
            y=metric,
            hue='model',
            marker='.',
            ax=ax,
            hue_order=models,
            palette=MODEL_COLOR_MAP,
            linewidth=1.5,
        )

        # Apply linestyles according to texture_map
        for line, model in zip(ax.get_lines(), models):
            line.set_linestyle('--' if model in TEXTURE_MAP else '-')

        # Add bert-baseline as a horizontal reference line
        bert_value = bert_metric_per_task.loc[bert_metric_per_task['task']
                                              == task, metric].values
        if len(bert_value) > 0:
            ax.axhline(
                y=bert_value[0],
                color=MODEL_COLOR_MAP['bert-baseline'],
                linestyle='--',
                linewidth=1.5,
                label='bert-baseline'
            )

        ax.set_title(task)
        ax.set_ylim(0, 1)
        ax.set_xlabel("")
        ax.legend().remove()

    # Mark the top 3 performance per task, any model x condition combination or bert-baseline
    rank_marker_map = {1: '*', 2: '^', 3: 'v'}
    for i, task in enumerate(tasks):
        ax = axes[i]
        task_subset = data[data['task'] == task].copy()
        bert_value = bert_metric_per_task.loc[bert_metric_per_task['task']
                                              == task, metric].values
        if len(bert_value) > 0:
            bert_row = pd.DataFrame([{
                'task': task,
                'model': 'bert-baseline',
                'condition': '0',
                metric: bert_value[0]
            }])
            task_subset = pd.concat([task_subset, bert_row], ignore_index=True)
        top3 = task_subset.nlargest(3, metric)

        previous_pos = [(0, 0)]
        for rank, (_, row) in enumerate(top3.iterrows(), start=1):
            x = COND_ORDER_SHORT.index(row['condition'])
            y = row[metric]
            model = row['model']

            ax.scatter(
                x=x,
                y=y,
                s=100,
                color=MODEL_COLOR_MAP.get(model, 'black'),
                marker=rank_marker_map[rank],
                zorder=10,
                edgecolor=MODEL_COLOR_MAP.get(model, 'black')
            )
            new_pos = (x,  y + 0.05)

            # Try to avoid overlapping text
            for prev in previous_pos:
                if abs(new_pos[0] - prev[0]) < 0.5 and abs(new_pos[1] - prev[1]) < 0.5:
                    new_pos = (x, y - 0.12)

            ax.text(
                new_pos[0], new_pos[1],
                f"{y:.2f}",
                ha='center',
                color='black',
                fontsize=10,
            )
            previous_pos.append(new_pos)

    handles_lines = [
        plt.Line2D([0, 1], [0, 0],
                   color=MODEL_COLOR_MAP.get(model, 'gray'),
                   linestyle='--' if model in TEXTURE_MAP or model=='bert-baseline' else '-',
                   linewidth=2)
        for model in models + ['bert-baseline']
    ]
    labels_lines = models + ['bert-baseline']

    handles_markers = [
        plt.Line2D([0], [0], color='white', marker=rank_marker_map[rank],
                   markeredgecolor='black', markersize=10, linestyle='None')
        for rank in [1, 2, 3]
    ]
    labels_markers = ['1st', '2nd', '3rd']

    if n_tasks == 3:
        fig.legend(
            handles=handles_lines + handles_markers,
            labels=labels_lines + labels_markers,
            title="Models & Top Ranking",
            loc='lower center',
            bbox_to_anchor=(0.5, -0.7),
            ncol=2,
            handlelength=2.5,
            columnspacing=1.0,
            title_fontsize='small'
        )
    else:
        fig.legend(
            handles=handles_lines + handles_markers,
            labels=labels_lines + labels_markers,
            title="Models & Top Ranking",
            loc='lower right',
            bbox_to_anchor=(0.96, 0.04),
            ncol=2,
            handlelength=2.5,
            columnspacing=1.0,
            title_fontsize='small'
        )
    # Remove empty axes if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title, y=0.98)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def make_ner_zero_shot_bar_plot(
        data: pd.DataFrame,
        title: str,
        save_path: str = None,
        metrics: list[str] = ["f1 overall - strict",
                              "f1 overall - partial", "f1_overall"]
):
    """ y axis metric, x axis models, metric grouped per model with error bars."""
    sns.set_style("darkgrid")
    # filter according to metrics
    data = data[data['metric'].isin(metrics)]

    # Ensure we only keep models present in the global model_order and preserve that order
    df = data.copy()
    # rename metrics for better display
    df['metric'] = df['metric'].replace(METRIC_RENAME_MAP)
    ordered_models = [m for m in MODEL_ORDER +
                      ['biomedbert-abstract'] if m in df['model'].unique()]
    # reorder data accordingly
    df['model'] = pd.Categorical(
        df['model'], categories=ordered_models, ordered=True)
    df = df.sort_values(['model', 'metric']).reset_index(drop=True)
    hue_order = [METRIC_RENAME_MAP.get(m, m) for m in metrics]

    # make width dependent on number of metrics
    n_metrics = len(hue_order)
    plt.figure(figsize=(n_metrics * 2, 6))
    # make color pallette based on metric_color_map
    ax = sns.barplot(
        data=df,
        x="model",
        y="mean",
        hue="metric",
        palette=[METRIC_COLOR_MAP.get(m, 'gray') for m in hue_order],
        order=ordered_models,
        hue_order=hue_order,
        dodge=True,
    )

    # iterate containers together with hue names so we can place CI errorbars aligned to bars
    for container, metric in zip(ax.containers, hue_order):
        metric_df = df[df['metric'] == metric].set_index('model')
        for i, bar in enumerate(container):
            model = ordered_models[i]
            row = metric_df.loc[model]
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_height()
            yerr_lower = float(row['mean']) - float(row['ci_lower'])
            yerr_upper = float(row['ci_upper']) - float(row['mean'])
            ax.errorbar(x, y, yerr=[[yerr_lower], [yerr_upper]],
                        fmt='none', color='black', capsize=2)

            # text at bottom of bar with mean value
            # ax.text(
            #         x, 0.02,s
            #         f"{float(row['mean']):.2f}",
            #         ha="center", va="center",
            #         color="black",
            #         fontsize=8, rotation=90, rotation_mode='anchor'
            #     )
    # set legend header to Metric
    handles = [plt.Rectangle((0, 0), 1, 1, color=color)
               for color in [METRIC_COLOR_MAP.get(m, 'gray') for m in hue_order]]
    ax.legend(handles, hue_order, title="Metric", loc="upper left",
              ncol=3, bbox_to_anchor=(0.01, 0.99))
    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel("Metric")
    # set y limit 0-1
    ax.set(ylim=(0, 1))
    ax.set_xticks(range(len(ordered_models)))
    ax.set_xticklabels(ordered_models, rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def make_ner_error_analysis_plot(data: pd.DataFrame, title: str, save_path: str = None):
    """Make a stacked bar plot showing error types per model for NER."""
    sns.set_style("darkgrid")

    # Remove correct errors
    data['error_type'] = data['error_type'].str.capitalize()

    # Fixed error types and colors
    error_types = ['Correct', 'Spurious', 'Missed', 'Incorrect']  # extend if needed
    color_palette = sns.color_palette("colorblind", n_colors=len(error_types))
    error_color_map = dict(zip(error_types, color_palette))

    # Model order
    all_models = data['model'].unique()
    ordered_models = [m for m in MODEL_ORDER +
                      ['biomedbert-abstract'] if m in all_models]

    # Append condition to model names if present
    if 'condition' in data.columns:
        for i, row in data.iterrows():
            if pd.notna(row['condition']):
                old_name = row['model']
                cond = COND_RENAME_MAP[row['condition']]
                new_name = f"{old_name}-{cond}"
                data.at[i, 'model'] = new_name
                # ordered_models is a Python list, so use list comprehension to replace entries
                ordered_models = [new_name if m == old_name else m for m in ordered_models]

    # for fine-tuned models, make sure Llama-3.1-8B-Instruct appears after all fine-tuned models
    if 'Llama-3.1-8B-Instruct-LST' in ordered_models:
        ordered_models.remove('Llama-3.1-8B-Instruct')
        # inject after LST model
        idx = ordered_models.index('Llama-3.1-8B-Instruct-LST') + 1
        ordered_models.insert(idx, 'Llama-3.1-8B-Instruct')

    plt.figure(figsize=(12, 6))
    bottoms = defaultdict(float)
    width = 0.8  # total width for stacked bars

    for error in error_types:
        # Select subset for this error type, reindex to keep consistent order
        subset = data[data["error_type"] == error].set_index(
            "model").reindex(ordered_models).fillna(0)
        counts = subset["count"].values

        plt.bar(
            ordered_models,
            counts,
            bottom=[bottoms[m] for m in ordered_models],
            color=error_color_map[error],
            width=width,
            label=error,
            edgecolor='white',  # subtle separation
        )

        # update stacking
        for m, c in zip(ordered_models, counts):
            bottoms[m] += c

        # add count labels on each segment
        for i, (m, c) in enumerate(zip(ordered_models, counts)):
            if c > 0:
                plt.text(
                    i,
                    bottoms[m] - c / 2,
                    str(int(c)),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black"
                )

    # Styling like few-shot variant
    ax = plt.gca()
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Model")
    plt.ylabel("Number of Errors (# Entities)")
    plt.title(title)
    plt.legend(title="Error Type", bbox_to_anchor=(1.02, 1),
               loc="upper left", title_fontsize="small")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def make_ner_few_shot_error_analysis_plot(data: pd.DataFrame, title: str, save_path: str = None):
    """ y axis: number of errors, x axis: models, grouped by condition, stacked by error type."""
    sns.set_style("darkgrid")

    data["error_type"] = data["error_type"].str.capitalize()
    data["condition"] = data["condition"].replace(COND_RENAME_MAP_SHORT)

    # Fixed error type order and colors
    error_order = ["Incorrect", "Spurious", "Missed", "Correct"]
    sns_palette = sns.color_palette("colorblind", n_colors=len(error_order))
    error_color_map = dict(zip(error_order, sns_palette))

    all_models = data["model"].unique().tolist()
    ordered_models = [m for m in MODEL_ORDER +
                      ['biomedbert-abstract'] if m in all_models]

    plt.figure(figsize=(12, 7))
    width = 0.18  # bar width per condition
    x_positions = np.arange(len(ordered_models))

    for c_idx, cond in enumerate(COND_ORDER_SHORT):
        cond_data = data[data["condition"] == cond]
        bottoms = defaultdict(float)
        offset = (c_idx - (len(COND_ORDER_SHORT) - 1) / 2) * width

        for error in error_order:
            subset = cond_data[cond_data["error_type"] == error].set_index(
                "model").reindex(ordered_models).fillna(0)
            counts = subset["count"].values
            plt.bar(
                x_positions + offset,
                counts,
                bottom=[bottoms[m] for m in ordered_models],
                color=error_color_map[error],
                width=width,
                label=error if c_idx == 0 else None,  # label once for legend
                alpha=0.9,
                linewidth=0.5,
            )

            for i, (m, c) in enumerate(zip(ordered_models, counts)):
                if c > 0:
                    base = bottoms[m]      # the bottom before adding c
                    y_pos = base + c / 2   # midpoint of this segment
                    
                    plt.text(
                        i + offset,
                        y_pos,
                        str(int(c)),
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="black"
                    )

            # update stacking
            for m, c in zip(ordered_models, counts):
                bottoms[m] += c

        # Add condition labels below bars
        for i, model in enumerate(ordered_models):
            plt.gca().text(
                i + offset,
                -0.5,
                cond,
                ha="center",
                va="top",
                fontsize=9,
            )

    plt.xticks(x_positions, ordered_models, rotation=45, ha="right")
    plt.xlabel("Model")
    plt.ylabel("Number of Errors (# Entities)")
    plt.title(title)
    plt.legend(title="Error Type", bbox_to_anchor=(1.02, 1),
               loc="upper left", title_fontsize="small")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def make_ner_few_shot_parallel_plot(data: pd.DataFrame, title: str, save_path: str = None, metrics: list[str] = ["f1 overall - strict", "precision overall - strict", "recall overall - strict"]):
    """ Plot few-shot NER performance across models and conditions with error bars."""
    sns.set_style("darkgrid")
    df = data.copy()

    # Renaming
    df['model'] = df['model'].apply(
        lambda x: 'bert-baseline' if 'bert' in x.lower() else x)
    df['condition'] = df['condition'].replace(COND_RENAME_MAP_SHORT)
    df['metric'] = df['metric'].replace(METRIC_RENAME_MAP)
    metrics = [METRIC_RENAME_MAP[m] for m in metrics]

    bert_df = df[(df['model'] == 'bert-baseline') & (df['condition'] == '0')]
    bert_metrics = {row['metric']: row['mean']
                    for _, row in bert_df.iterrows()}

    # Filtering
    df = df[~df['model'].isin(['tuned', 'bert-baseline'])]

    # Ordering
    models = [m for m in MODEL_ORDER if m in df['model'].unique()]
    df['model'] = pd.Categorical(df['model'], categories=models, ordered=True)
    df['condition'] = pd.Categorical(
    df['condition'], categories=COND_ORDER_SHORT, ordered=True)
    # Layout: roughly square grid
    ncols = len(metrics)
    fig, axes = plt.subplots(nrows=1, ncols=ncols,
                             figsize=(5 * ncols, 4), sharey=False)
    axes = np.array(axes).flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        # Plot the main lines
        sns.lineplot(
            data=df[df["metric"] == metric],
            x='condition',
            y='mean',
            hue='model',
            marker='.',
            ax=ax,
            hue_order=models,
            palette=MODEL_COLOR_MAP,
            linewidth=1.5,
        )

        # Apply linestyles according to texture_map
        for line, model in zip(ax.get_lines(), models):
            if model in TEXTURE_MAP:
                line.set_linestyle('--')
            else:
                line.set_linestyle('-')

        # Add bert-baseline as a horizontal reference line
        ax.axhline(
            y=bert_metrics[metric],
            color=MODEL_COLOR_MAP['bert-baseline'],
            linestyle='--',
            linewidth=1.5,
            label='bert-baseline'
        )
        ax.set_ylim(0, 1)
        ax.set_ylabel(metric)
        ax.set_xlabel("")
        ax.legend().remove()

    rank_marker_map = {1: '*', 2: '^', 3: 'v'}

    for i, metric in enumerate(metrics):
        ax = axes[i]
        rank_df = pd.concat([
            df[['model', 'condition', 'mean', 'metric']],
            bert_df[['model', 'condition', 'mean', 'metric']]
        ])
        # Filter for current metric
        rank_df = rank_df[rank_df['metric'] == metric]

        top3 = rank_df.nlargest(3, 'mean')
        previous_pos = [(0, 0)]
        for rank, (_, row) in enumerate(top3.iterrows(), start=1):
            x = COND_ORDER_SHORT.index(row['condition'])
            y = row['mean']
            model = row['model']

            ax.scatter(
                x=x,
                y=y,
                s=100,
                color=MODEL_COLOR_MAP.get(model, 'black'),
                marker=rank_marker_map[rank],
                zorder=10,
                edgecolor=MODEL_COLOR_MAP.get(model, 'black')
            )
            new_pos = (x,  y + 0.05)

            # if the new position (x any y) is too close to the previous, put number below
            for prev in previous_pos:
                if abs(new_pos[0] - prev[0]) < 0.5 and abs(new_pos[1] - prev[1]) < 0.5:
                    new_pos = (x, y - 0.12)

            ax.text(
                new_pos[0], new_pos[1],
                f"{y:.2f}",
                ha='center',
                color='black',
                fontsize=10,
            )
            previous_pos.append(new_pos)

    handles_lines = [
        plt.Line2D([0, 1], [0, 0],
                   color=MODEL_COLOR_MAP.get(model, 'gray'),
                   linestyle='--' if model in TEXTURE_MAP else '-',
                   linewidth=2)
        for model in models + ['bert-baseline']
    ]
    labels_lines = models + ['bert-baseline']

    handles_markers = [
        plt.Line2D([0], [0], color='white', marker=rank_marker_map[rank],
                   markeredgecolor='black', markersize=10, linestyle='None')
        for rank in [1, 2, 3]
    ]
    labels_markers = ['1st', '2nd', '3rd']
    # place legend centered below the plot, spread into 4 columns
    fig.legend(
        handles=handles_lines + handles_markers,
        labels=labels_lines + labels_markers,
        title="Models & Top Ranking",
        loc='lower center',
        bbox_to_anchor=(0.5, -0.35),
        ncol=4,
        handlelength=2.5,
        columnspacing=1.0,
    )

    # Remove empty axes if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title, y=0.98)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def make_ift_performance_plot(
    data: pd.DataFrame,
    best_model: pd.DataFrame,
    title: str,
    save_path: str = None,
    metric: str = "f1-weighted",
):
    """
    Vertical bar plots per task in a grid layout.
    Each subplot: x = model (vertical bars), y = metric.
    """
    sns.set_style("darkgrid")
    # Combine data and mark sources
    combined = pd.concat(
        [data.assign(source="IFT/Baseline"), best_model.assign(source="Best")],
        ignore_index=True
    )

    tasks = sorted(combined["task"].unique())
    n_tasks = len(tasks)

    if n_tasks == 1:
        ncols, nrows = 1, 1
    elif n_tasks <= 4:
        ncols, nrows = n_tasks, 1
    elif n_tasks <= 8:
        ncols, nrows = 4, 2
    else:
        ncols, nrows = 5, int(np.ceil(n_tasks / 5))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5 * ncols, 4 * nrows),
        sharey=False
    )

    # Flatten axes for consistent indexing
    if n_tasks == 1:
        axes = np.array([axes])  # make iterable
    axes = axes.flatten()

    for i, task in enumerate(tasks):
        ax = axes[i]
        subset = combined[combined["task"] == task].copy()

        # Find best model info
        best_idx = subset[subset["source"] == "Best"].index[0]
        condition = subset.loc[best_idx, "condition"].lstrip("selected_")
        new_name = f'{subset.loc[best_idx, "model"]}-{condition}'
        subset.loc[best_idx, "model"] = new_name

        fixed_order = [
            new_name,
            'Llama-3.1-8B-Instruct-IFT',
            'Llama-3.1-8B-Instruct',
            'bert-baseline'
        ]
        if 'Llama-3.1-8B-Instruct-LST' in subset['model'].values:
            fixed_order.insert(2, 'Llama-3.1-8B-Instruct-LST')

        # Palette setup
        palette = {}
        for j, m in enumerate(fixed_order):
            if j == 0:
                palette[m] = MODEL_COLOR_MAP.get(
                    m.rstrip(condition).rstrip('-'), '#cccccc')
            else:
                palette[m] = MODEL_COLOR_MAP.get(m, '#cccccc')

        sns.barplot(
            data=subset,
            x="model",
            y=metric,
            palette=palette,
            order=fixed_order,
            ax=ax,
            errorbar=None,
        )

        ax.set_xticklabels(fixed_order, rotation=45, ha="right", fontsize=9)

        # Add value annotations and error bars
        for idx, patch in enumerate(ax.patches):
            model = fixed_order[idx]
            value = subset.loc[subset['model'] == model, metric].iloc[0]
            x_center = patch.get_x() + patch.get_width() / 2
            y_bottom = patch.get_y()
            y_top = patch.get_height()

            ax.text(
                x_center, y_bottom + 0.01, f"{value:.2f}",
                ha="center", va="bottom", fontsize=9, color="black"
            )

            # error bars
            row = subset[subset['model'] == model].iloc[0]
            if 'lower' in row and 'upper' in row:
                yerr_lower = float(row[metric]) - float(row['lower'])
                yerr_upper = float(row['upper']) - float(row[metric])
                ax.errorbar(
                    x_center, y_top, yerr=[[yerr_lower], [yerr_upper]],
                    fmt='none', color='black', capsize=5
                )

        # Annotate best model condition
        patch = ax.patches[0]
        x_center = patch.get_x() + patch.get_width() / 2
        y_center = patch.get_height() / 2
        condition_rename = {
            'zero_shot': '0-shot',
            '1shot': '1-shot',
            '3shot': '3-shot',
            '5shot': '5-shot',
        }
        ax.text(
            x_center, y_center, condition_rename.get(condition, condition),
            ha="center", va="center",
            color='black', fontsize=9, fontweight="bold"
        )

        if 'Llama-3.1-8B-Instruct-LST' in subset['model'].values:
            patch_nr = 3
        else:
            patch_nr = 2
        patch = ax.patches[patch_nr]
        x_center = patch.get_x() + patch.get_width() / 2
        y_center = patch.get_height() / 2
        ax.text(
            x_center, y_center, '0-shot',
            ha="center", va="center",
            color='black', fontsize=9, fontweight="bold"
        )
        if len(tasks) > 1:
            ax.set_title(task.replace("_", " ").title(), fontsize=12)
        ax.set_xlabel("")
        ax.set_xticks([])
        ax.set_ylabel("")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Remove unused axes if fewer tasks
    for j in range(n_tasks, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.subplots_adjust(top=0.94, bottom=0.06)
    if len(tasks) <= 4:
        fig.suptitle(title, y=1.05)
    else:
        fig.suptitle(title)

    best_models = best_model['model'].unique().tolist()
    fine_tuned_models = fixed_order[1:]
    present_models = best_models + fine_tuned_models
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=MODEL_COLOR_MAP.get(m, '#cccccc'))
        for m in present_models
    ]
    if len(tasks) > 4:
        fig.legend(
            handles, present_models, title="Model",
            loc='lower right', bbox_to_anchor=(0.98, 0.1), ncol=4
        )
    # add legend to the right of the plot if few tasks, to avoid overlapping
    else:
        fig.legend(
            handles, present_models, title="Model",
            loc='center right', bbox_to_anchor=(1.5, 0.5), ncol=1
        )

    # add y label to each row
    for row in range(nrows):
        axes[row * ncols].set_ylabel(METRIC_RENAME_MAP.get(metric, metric))

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def ift_box_plot(ift_df: pd.DataFrame, title: str, save_path: str = None):
    # rename all models in best_df to 'Best ICL'
    sns.set_style("darkgrid")

    model_order = ['gpt-4o-2024-08-06', 'Llama-3.1-8B-Instruct-IFT', 'Llama-3.1-8B-Instruct', 'bert-baseline']
    ift_df['model'] = pd.Categorical(
        ift_df['model'], categories=model_order, ordered=True)

    plt.figure(figsize=(5, 4))
    ax = sns.boxplot(
        data=ift_df,
        x='model',
        y='f1-weighted',
        order=model_order,
        palette=[MODEL_COLOR_MAP.get(m, 'white') for m in model_order],
        hue='model',
    )
    # add performance scatter points, on topt
    sns.stripplot(
        data=ift_df,
        x="model",
        y="f1-weighted",
        color="black",
        size=3,
        alpha=0.6,
        jitter=True,
    )

    # rotate x-axis
    ax.set_xticklabels(model_order, rotation=25, ha="right", fontsize=10)
    ax.set_title(title, pad=12)
    plt.xlabel("Model")
    plt.ylabel("f1-weighted")
    plt.ylim(0, 1)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def size_plot(data: pd.DataFrame, title: str, save_path: str = None):
    """
    Scatter plot with:
        x-axis = model_size (ordered numerically)
        y-axis = metric (performance)
        color = model
        marker shape = condition
    """
    sns.set_style("darkgrid")

    model_order = [
        'bert-baseline',
        'Meta-Llama-3-8B-Instruct',
        'Med-LLaMA3-8B',
        'Llama-3.1-8B-Instruct',
#         'Llama-3.1-8B-Instruct-IFT',
        'Llama-2-13b-chat-hf',
        'MeLLaMA-13B-chat',
        'gemma-3-27b-it',
        'medgemma-27b-text-it',
        'gpt-4o-mini',
        'Llama-2-70b-chat-hf',
        'MeLLaMA-70B-chat',
        'gpt-4o-2024-08-06',
    ]

    condition_order = [
        'zero-shot',
        'few-shot',
        'fine-tuned',
    ]

    # Keep only models in our desired order and ensure model is categorical
    data_sorted = data[data["model"].isin(model_order)].copy()
    data_sorted["model"] = pd.Categorical(data_sorted["model"], categories=model_order, ordered=True)
    data_sorted = data_sorted.sort_values("model")

    # enforce condition order and make categorical so hue respects order
    data_sorted["condition"] = pd.Categorical(data_sorted["condition"],
                                             categories=condition_order,
                                             ordered=True)

    plt.figure(figsize=(10, 6))

    ax = sns.boxplot(
        data=data_sorted,
        x="model",
        y="metric",
        hue="condition",
        hue_order=condition_order,
    )

    plt.title(title)
    plt.xlabel("Model Size (B parameters)")
    plt.ylabel("Metric")

    # Prepare model_size labels in the same order as plotted models
    df_model_only = data_sorted.drop_duplicates(subset=["model", "model_size"]).set_index("model")
    models_present = [m for m in model_order if m in df_model_only.index]
    sizes = df_model_only.reindex(models_present)["model_size"].tolist()

    ax.legend(title="Condition", bbox_to_anchor=(1.02, 1), loc="upper left")

    model_labels = [
        r"$\mathrm{bert\text{-}baseline\ (}\mathbf{0.11B}\mathrm{)}$",
        r"$\mathrm{Meta\text{-}Llama\text{-}3\text{-}8B\text{-}Instruct\ (}\mathbf{8B}\mathrm{)}$",
        r"$\mathrm{Med\text{-}LLaMA3\text{-}8B\ (}\mathbf{8B}\mathrm{)}$",
        r"$\mathrm{Llama\text{-}3.1\text{-}8B\text{-}Instruct\ (}\mathbf{8B}\mathrm{)}$",
        r"$\mathrm{Llama\text{-}2\text{-}13b\text{-}chat\text{-}hf\ (}\mathbf{13B}\mathrm{)}$",
        r"$\mathrm{MeLLaMA\text{-}13B\text{-}chat\ (}\mathbf{13B}\mathrm{)}$",
        r"$\mathrm{gemma\text{-}3\text{-}27b\text{-}it\ (}\mathbf{27B}\mathrm{)}$",
        r"$\mathrm{medgemma\text{-}27b\text{-}text\text{-}it\ (}\mathbf{27B}\mathrm{)}$",
        r"$\mathrm{gpt\text{-}4o\text{-}mini\ (}\mathbf{\sim30B}\mathrm{)}$",
        r"$\mathrm{Llama\text{-}2\text{-}70b\text{-}chat\text{-}hf\ (}\mathbf{70B}\mathrm{)}$",
        r"$\mathrm{MeLLaMA\text{-}70B\text{-}chat\ (}\mathbf{70B}\mathrm{)}$",
        r"$\mathrm{gpt\text{-}4o\text{-}2024\text{-}08\text{-}06\ (}\mathbf{\sim200B}\mathrm{)}$",
    ]
    ax.set_xticks(range(len(model_labels)))
    ax.set_xticklabels(model_labels, rotation=45, ha="right")


    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


def medical_vs_general_plot(data: pd.DataFrame, title: str, save_path: str = None):
    """
    Compare medical vs general models using seaborn boxplots.
    Each condition gets its own subplot.
    Medical models are hatched and have a legend.
    Colors come from MODEL_COLOR_MAP.
    """
    sns.set_style("darkgrid")

    model_pairs = [
        ['Meta-Llama-3-8B-Instruct', 'Med-LLaMA3-8B'],
        ['Llama-2-13b-chat-hf', 'MeLLaMA-13B-chat'],
        ['Llama-2-70b-chat-hf', 'MeLLaMA-70B-chat'],
        ['gemma-3-27b-it', 'medgemma-27b-text-it'],
    ]

    conditions = ['zero-shot', 'few-shot']
    fig, axes = plt.subplots(1, len(conditions), figsize=(10, 5), sharey=True)
    if len(conditions) == 1:
        axes = [axes]

    for ax, condition in zip(axes, conditions):
        data_cond = data[data['condition'] == condition].copy()

        # Flatten model_pairs for ordering
        flat_models = [m for pair in model_pairs for m in pair]
        data_cond['model'] = pd.Categorical(data_cond['model'], categories=flat_models, ordered=True)

        # Plot seaborn boxplot
        sns.boxplot(
            data=data_cond,
            x='model',
            y='metric',
            ax=ax,
            palette=MODEL_COLOR_MAP,
            dodge=False
        )

        # Add hatching for medical models
        for i, patch in enumerate(ax.patches):
            model_name = flat_models[i % len(flat_models)]
            if model_name in TEXTURE_MAP:
                patch.set_hatch('//')
                patch.set_edgecolor('white')
                patch.set_linewidth(1.5)

        ax.set_xticklabels(flat_models, rotation=45, ha='right')
        ax.set_title(condition.title())
        ax.set_xlabel("Model")
        ax.set_ylabel("Metric")

    # Add a custom legend for medical models
    general_patch = mpatches.Patch(facecolor='white', edgecolor='black', label='General Model')
    medical_patch = mpatches.Patch(facecolor='black', edgecolor='white', hatch='//', label='Medical Model')
    fig.legend(handles=[general_patch, medical_patch], loc='upper right')

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300)
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
