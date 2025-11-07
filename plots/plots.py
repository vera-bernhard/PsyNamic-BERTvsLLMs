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
model_color_map = {
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
    'Llama-3.1-8B-Instruct-IFT': "#ca9161"
}

metric_color_map = {
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

texture_map = {
    'MeLLaMA-70B-chat': 'hatch',
    'MeLLaMA-13B-chat': 'hatch',
    'Med-LLaMA3-8B': 'hatch',
    'medgemma-27b-text-it': 'hatch',
}

model_order = [
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
    'bert-baseline',
]

condition_order = ["zero_shot", "selected_1shot",
                   "selected_3shot", "selected_5shot"]

cond_rename_map = {
    'zero_shot': '0',
    'selected_1shot': '1',
    'selected_3shot': '3',
    'selected_5shot': '5',
}
condition_renamed_order = ['0', '1', '3', '5']


def make_performance_plot(data: dict, task: str, save_path: str = None, metrics_col: str = 'metrics') -> None:
    if isinstance(data, dict):
        rows = []
        for model, values in data.items():
            # allow bert baselines even if not in model_order
            if model not in model_order and 'bert' not in model.lower():
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
        model_colors[m] = model_color_map.get(
            m, default_palette[i % len(default_palette)])
    # special-case bert
    for m in unique_models:
        if 'bert' in m.lower():
            model_colors[m] = model_color_map['bert-baseline']

    metrics = df_metrics['metric'].unique()
    num_metrics = len(metrics)
    num_models = len(unique_models)
    fig_width = num_metrics * max(4, num_models * 1.5)
    fig, axes = plt.subplots(
        1, num_metrics, figsize=(fig_width, 7), sharey=True)
    plt.subplots_adjust(wspace=0.15)
    plt.suptitle(
        f"Zero-Shot Comparison for {task}", fontsize=16, fontweight='semibold', y=0.98)

    if num_metrics == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        ax = axes[i]
        df_sub = df_metrics[df_metrics['metric'] == metric].copy()

        ordered_models = [
            m for m in model_order if m in df_sub['model'].values]
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
    model_colors = {model: model_color_map.get(model, sns.color_palette("tab10")[i % 10])
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


def make_zero_shot_scatter_plot(data: pd.DataFrame, title: str, save_path: str = None) -> None:
    """ x axis tasks, y axis performance, x axis grouped by model with dots with error bars next to each other."""
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # determine task order and model order (respect global model_order first)
    tasks = list(data['task'].unique())
    unique_models = list(data['model'].unique())
    ordered_models = [m for m in model_order if m in unique_models]
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
        color = model_color_map.get(model, sns.color_palette("tab10")[i % 10])
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
    # remove any models which are not in model order
    df = df[df['model'].isin(model_order)]

    # Set up colors for conditions (keep order consistent)
    unique_conditions = [
        c for c in condition_order if c in df['condition'].unique()]
    # Add any other conditions not in the order list
    unique_conditions += [c for c in df['condition'].unique()
                          if c not in unique_conditions]
    palette = sns.color_palette("tab10", len(unique_conditions))
    condition_colors = dict(zip(unique_conditions, palette))

    # Make sure models are ordered according to model_order
    df['model'] = pd.Categorical(df['model'],
                                 categories=model_order,
                                 ordered=True)
    # Make chart wider (increased figsize)
    fig, ax = plt.subplots(figsize=(28, 7))  # wider plot

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
    ax.set_title(f"In-context Learning Comparison for {task}")
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
            # ax.text(x, row["ci_lower"] - 0.03,
            #         f"{row['ci_lower']:.3f}", ha="center", va="bottom", color="black", fontsize=8)
            # CI upper
            # ax.text(x, row["ci_upper"] + 0.01,
            #         f"{row['ci_upper']:.3f}", ha="center", va="bottom", color="black", fontsize=8)
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
    # Ensure we only keep models present in the global model_order and preserve that order
    df = data.copy()
    df = df[df['model'].isin(model_order)]
    ordered_models = [m for m in model_order if m in df['model'].unique()]

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
        palette=[model_color_map.get(model, "#cccccc") for model in ordered_models]
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
    data = data[data['model'].isin(model_order)]
    df_pivot = data.pivot(index="model", columns="task", values="performance")
    tasks = df_pivot.columns.tolist()
    num_tasks = len(tasks)

    angles = np.linspace(0, 2 * np.pi, num_tasks, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # order models according to model_order
    models_in_df = list(df_pivot.index)
    ordered_models = [m for m in model_order if m in models_in_df]
    ordered_models += [m for m in models_in_df if m not in ordered_models]

    for model, row in df_pivot.iterrows():
        values = row.tolist()
        values += values[:1]
        color = model_color_map.get(model, sns.color_palette(
            "tab10")[ordered_models.index(model) % 10])
        linestyle = '--' if model in texture_map else '-'
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
    data['model'] = data['model'].apply(
        lambda x: 'bert-baseline' if 'bert' in x.lower() else x)

    data['condition'] = data['condition'].apply(
        lambda x: cond_rename_map.get(x, x))

    skip_models = ['tuned', 'bert-baseline', 'Med-LLaMA3-8B', 'gemma-3-27b-it']
    data = data[~data['model'].isin(skip_models)]
    tasks = data['task'].unique().tolist()
    models = data['model'].unique().tolist()
    models = [m for m in model_order if m in models]

    data = data[data['task'].isin(tasks)]
    data = data[data['model'].isin(models)]

    data['model'] = pd.Categorical(
        data['model'], categories=models, ordered=True)
    data['condition'] = pd.Categorical(
        data['condition'], categories=condition_renamed_order, ordered=True)

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
        g.fig.suptitle(title, fontsize=16, fontweight='semibold',
                       color="#222222", y=0.98)
        g.fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        g.fig.tight_layout()

    if save_path:
        g.fig.savefig(save_path, bbox_inches='tight')
        plt.close(g.fig)
    else:
        plt.show()


def make_few_shot_delta_plot(data: pd.DataFrame, title: str, save_path: str = None, metric: str = "f1-weighted"):
    """ - y = tasks
        - x = condition
        - x groupe by model

        multiplot models * tasks with lines with points for each condition
    """
    sns.set_style("darkgrid")
    data['model'] = data['model'].apply(
        lambda x: 'bert-baseline' if 'bert' in x.lower() else x)

    condition_order = ['1', '3', '5']
    data['condition'] = data['condition'].apply(
        lambda x: cond_rename_map.get(x, x))
    skip_models = ['tuned', 'bert-baseline',
                   'Med-LLaMA3-8B']  # , 'gemma-3-27b-it']
    data = data[~data['model'].isin(skip_models)]
    tasks = data['task'].unique().tolist()
    data = data[data['task'].isin(tasks)]
    models = data['model'].unique().tolist()
    models = [m for m in model_order if m in models]
    data = data[data['model'].isin(models)]

    # Calculate relative percentage improvements compared to zero-shot and then normalize
    for model in models:
        for task in tasks:
            baseline = data[(data['model'] == model) & (
                data['task'] == task) & (data['condition'] == '0')]
            for cond in condition_order:
                current = data[(data['model'] == model) & (
                    data['task'] == task) & (data['condition'] == cond)]
                if len(current) > 0:
                    try:
                        # calculate relative (in %) improvement over zero-shot, if got worse its negative
                        improvement = current[metric].values[0] - \
                            baseline[metric].values[0]
                        # overwrite condition value with improvement
                        data.loc[current.index[0], metric] = improvement
                    except IndexError:
                        # no baseline found, skip
                        breakpoint()

    # remove all zero-shot rows as they are now baselines
    data = data[data['condition'] != '0']
    # add color column based on positive/negative improvement
    data['color'] = data[metric].apply(lambda x: 'green' if x > 0 else 'red')

    models = data['model'].unique().tolist()
    models = [m for m in model_order if m in models]


    data['model'] = pd.Categorical(
        data['model'], categories=models, ordered=True)
    data['condition'] = pd.Categorical(
        data['condition'], categories=condition_renamed_order[1:], ordered=True)
    ncols = len(models)
    nrows = len(tasks)

    A4_width, A4_height = 8.27, 11.69
    nrows = len(tasks)
    ncols = len(models)

    g = sns.FacetGrid(
        data,
        col="model",
        row="task",
        sharey=True,
        sharex=True,
        # height=A4_height / nrows,
        # aspect=(A4_width / ncols) / (A4_height / nrows)
    )
    g.map_dataframe(sns.barplot, x="condition", y=metric, hue="color",
                    dodge=False, palette={"green": "green", "red": "red"})

    # Remove numbers and spans
    for row_idx, row_axes in enumerate(g.axes):
        for col_idx, ax in enumerate(row_axes):
            # Clean up unnecessary spines
            for spine in ["right", "top"]:
                ax.spines[spine].set_visible(False)

            # Only show y-axis ticks and labels on the first column
            if col_idx == 0:
                ax.tick_params(axis="y", which="both", left=True, labelleft=True)
                ax.spines["left"].set_visible(True)
                ax.set_ylabel("")
            else:
                ax.tick_params(axis="y", which="both", left=False, labelleft=False)
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

    # # Task names to the left
    if len(tasks) > 1:
        max_task_len = max((len(str(t)) for t in tasks), default=0)
        labelpad_left = 35 + max(0, (max_task_len - 10) * 3)
        for row_idx, task_name in enumerate(tasks):
            left_ax = axes[row_idx, 0]
            left_ax.set_ylabel(task_name, rotation=0,
                            labelpad=labelpad_left, va='center')
            for cond in range(1, len(models)):
                axes[row_idx, cond].set_ylabel("Δ F1")

    # Remove any subtitles/titles inside each subplot and the legend
    for ax in axes.flatten():
        ax.set_title("")
        for txt in list(ax.texts):
            ax.texts.remove(txt)

    # Add overall title if provided and adjust layout
    if title:
        g.fig.suptitle(title, fontsize=16, fontweight='semibold',
                       color="#222222", y=0.98)
        g.fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        g.fig.tight_layout()

    if save_path:
        g.fig.savefig(save_path, bbox_inches='tight')
        plt.close(g.fig)
    else:
        plt.show()


def make_few_shot_avg_plot(data: pd.DataFrame, title: str, save_path: str = None, metric: str = "f1-weighted"):
    """ Plot average few-shot performance across tasks for each model and condition, including error bars for stddev."""

    rename_map = {
        'zero_shot': 'zero-shot',
        'selected_1shot': '1-shot',
        'selected_3shot': '3-shot',
        'selected_5shot': '5-shot',
    }

    # Remove models
    data['model'] = data['model'].apply(
        lambda x: 'bert-baseline' if 'bert' in x.lower() else x)
    skip_models = ['tuned', 'bert-baseline']  # , 'gemma-3-27b-it']
    data = data[~data['model'].isin(skip_models)]
    data['condition'] = data['condition'].replace(rename_map)

    # average across tasks
    df_avg = (
        data.groupby(['model', 'condition'])[metric]
        .agg(['mean', 'std'])
        .reset_index()
    )

    # order models and conditions
    df_avg = df_avg[df_avg['model'].isin(model_order)]
    df_avg['model'] = pd.Categorical(
        df_avg['model'], categories=model_order, ordered=True)

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


def make_few_shot_parallel_plot(
    data: pd.DataFrame,
    title: str,
    save_path: str = None,
    metric: str = "f1-weighted"
):
    """Make parallel plots: one subplot per task, x=condition, lines=each model, reference line for bert-baseline."""
    sns.set_style("darkgrid")

    # Simplify model names if needed
    data['model'] = data['model'].apply(
        lambda x: 'bert-baseline' if 'bert' in x.lower() else x)

    # Extract bert-baseline performance per task
    bert_metric_per_task = data[data['model']
                                == 'bert-baseline'][['task', metric]]

    # Skip unwanted models in the main lines
    skip_models = ['tuned', 'bert-baseline']
    data = data[~data['model'].isin(skip_models)]

    # Normalize condition names and order
    rename_map = {
        'zero_shot': '0',
        'selected_1shot': '1',
        'selected_3shot': '3',
        'selected_5shot': '5',
    }
    data['condition'] = data['condition'].replace(rename_map)
    condition_order = ['0', '1', '3', '5']

    # Filter and order models
    models = [m for m in model_order if m in data['model'].unique()]
    data['model'] = pd.Categorical(
        data['model'], categories=models, ordered=True)
    data['condition'] = pd.Categorical(
        data['condition'], categories=condition_order, ordered=True)

    tasks = sorted(data['task'].unique())
    n_tasks = len(tasks)

    # Layout: roughly square grid
    ncols = 4
    nrows = 5

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(4 * ncols, 3 * nrows), sharey=True)
    axes = np.array(axes).flatten()

    for i, task in enumerate(tasks):
        ax = axes[i]
        subset = data[data['task'] == task]

        # Plot the main lines
        sns.lineplot(
            data=subset,
            x='condition',
            y=metric,
            hue='model',
            marker='.',
            ax=ax,
            hue_order=models,
            palette=model_color_map,
            linewidth=1.5,
        )

        # Apply linestyles according to texture_map
        for line, model in zip(ax.get_lines(), models):
            if model in texture_map:
                line.set_linestyle('--')
            else:
                line.set_linestyle('-')

        # Add bert-baseline as a horizontal reference line
        bert_value = bert_metric_per_task.loc[bert_metric_per_task['task']
                                              == task, metric].values
        if len(bert_value) > 0:
            ax.axhline(
                y=bert_value[0],
                color=model_color_map.get('bert-baseline', 'gray'),
                linestyle='--',
                linewidth=1.5,
                label='bert-baseline'
            )

        ax.set_title(task)
        ax.set_ylim(0, 1)
        ax.set_xlabel("")
        ax.legend().remove()

    rank_marker_map = {1: '*', 2: '^', 3: 'v'}

    for i, task in enumerate(tasks):
        ax = axes[i]

        # Include BERT in top 3 evaluation if it qualifies
        task_subset = data[data['task'] == task].copy()
        bert_value = bert_metric_per_task.loc[bert_metric_per_task['task']
                                              == task, metric].values
        if len(bert_value) > 0:
            bert_row = pd.DataFrame([{
                'task': task,
                'model': 'bert-baseline',
                'condition': '0',  # or whichever condition you want to use
                metric: bert_value[0]
            }])
            task_subset = pd.concat([task_subset, bert_row], ignore_index=True)
        top3 = task_subset.nlargest(3, metric)

        previous_pos = [(0, 0)]
        for rank, (_, row) in enumerate(top3.iterrows(), start=1):
            x = condition_order.index(row['condition'])
            y = row[metric]
            model = row['model']

            ax.scatter(
                x=x,
                y=y,
                s=100,
                color=model_color_map.get(model, 'black'),
                marker=rank_marker_map[rank],
                zorder=10,
                edgecolor=model_color_map.get(model, 'black')
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
                   color=model_color_map.get(model, 'gray'),
                   linestyle='--' if model in texture_map else '-',
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

    # two-column legend placed lower to avoid overlapping the plot
    fig.legend(
        handles=handles_lines + handles_markers,
        labels=labels_lines + labels_markers,
        title="Models & Top Ranking",
        loc='lower right',
        bbox_to_anchor=(0.96, 0.04),
        ncol=2,
        handlelength=2.5,
        columnspacing=1.0,
    )

    # Remove empty axes if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title, fontsize=16, fontweight='semibold',
                 color="#222222", y=0.98)
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
        metrics: list[str] = ["f1 overall - strict", "f1 overall - partial", "f1_entity_type"]
):
    """ y axis metric, x axis models, metric grouped per model with error bars."""
    # filter according to metrics
    data = data[data['metric'].isin(metrics)]
    rename_metrics = {
        "f1_entity_type": "F1 Strict",
        "f1 overall - strict": "F1 BIO - Strict",
        "f1 overall - partial": "F1 BIO - Partial",
        "precision overall - partial": "Precision BIO - Partial",
        "precision overall - strict": "Precision BIO - Strict",
        "precision_entity_type": "Precision Strict",
        "recall overall - partial": "Recall BIO - Partial",
        "recall overall - strict": "Recall BIO - Strict",
        "recall_entity_type": "Recall Strict",
    }

    # Ensure we only keep models present in the global model_order and preserve that order
    df = data.copy()
    # rename metrics for better display
    df['metric'] = df['metric'].replace(rename_metrics)
    ordered_models = [m for m in model_order +
                      ['biomedbert-abstract'] if m in df['model'].unique()]
    # reorder data accordingly
    df['model'] = pd.Categorical(
        df['model'], categories=ordered_models, ordered=True)
    df = df.sort_values(['model', 'metric']).reset_index(drop=True)
    hue_order = [rename_metrics.get(m, m) for m in metrics]
    
    # make width dependent on number of metrics
    n_metrics = len(hue_order)
    plt.figure(figsize=(n_metrics * 2, 6))
    # make color pallette based on metric_color_map
    ax = sns.barplot(
        data=df,
        x="model",
        y="mean",
        hue="metric",
        palette=[metric_color_map.get(m, 'gray') for m in hue_order],
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
               for color in [metric_color_map.get(m, 'gray') for m in hue_order]]
    labels = [rename_metrics.get(m, m) for m in hue_order]
    ax.legend(handles, labels, title="Metric", loc="upper left",
              ncol=3, bbox_to_anchor=(0.01, 0.99))
    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel("F1 Score")
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
    data = data[data['error_type'] != 'correct'].copy()
    data['error_type'] = data['error_type'].str.capitalize()

    # Fixed error types and colors
    error_types = ['Incorrect', 'Partial', 'Spurious', 'Missed']  # extend if needed
    color_palette = sns.color_palette("colorblind", n_colors=len(error_types))
    error_color_map = dict(zip(error_types, color_palette))

    # Model order
    all_models = data['model'].unique()
    ordered_models = [m for m in model_order + ['biomedbert-abstract'] if m in all_models]

    plt.figure(figsize=(12, 6))
    bottoms = defaultdict(float)
    width = 0.8  # total width for stacked bars

    for error in error_types:
        # Select subset for this error type, reindex to keep consistent order
        subset = data[data["error_type"] == error].set_index("model").reindex(ordered_models).fillna(0)
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
    plt.title(title, fontsize=14, fontweight="semibold")
    plt.legend(title="Error Type", bbox_to_anchor=(1.02, 1), loc="upper left", title_fontsize="small")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def make_ner_few_shot_error_analysis_plot(data: pd.DataFrame, title: str, save_path: str = None):
    data = data[data["error_type"] != "correct"].copy()
    data["error_type"] = data["error_type"].str.capitalize()

    # Map conditions to consistent labels and order
    cond_rename_map = {
        "zero_shot": "0",
        "selected_1shot": "1",
        "selected_3shot": "3",
        "selected_5shot": "5"
    }
    data["condition"] = data["condition"].replace(cond_rename_map)
    condition_order = ["0", "1", "3", "5"]

    # Fixed error type order and colors
    error_types = ["Incorrect", "Partial", "Spurious", "Missed"]  # extend if you have more types
    sns_palette = sns.color_palette("colorblind", n_colors=len(error_types))
    error_color_map = dict(zip(error_types, sns_palette))

    all_models = data["model"].unique().tolist()
    ordered_models = [m for m in model_order + ['biomedbert-abstract'] if m in all_models]

    plt.figure(figsize=(12, 7))
    width = 0.18  # bar width per condition
    x_positions = np.arange(len(ordered_models))

    for c_idx, cond in enumerate(condition_order):
        cond_data = data[data["condition"] == cond]
        bottoms = defaultdict(float)
        offset = (c_idx - (len(condition_order) - 1) / 2) * width

        for error in error_types:
            subset = cond_data[cond_data["error_type"] == error].set_index("model").reindex(ordered_models).fillna(0)
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

            # for i, (m, c) in enumerate(zip(ordered_models, counts)):
            #     if c > 0:
            #         plt.text(
            #             i + offset,
            #             bottoms[m] + c/2 - c,  # middle of this segment
            #             str(int(c)),
            #             ha="center",
            #             va="center",
            #             fontsize=9,
            #             color="black"
            #         )

            # update stacking
            for m, c in zip(ordered_models, counts):
                bottoms[m] += c

        # Add condition labels below bars
        for i, model in enumerate(ordered_models):
            plt.gca().text(
                i + offset,
                -0.5,  # slightly below x-axis
                cond,
                ha="center",
                va="top",
                fontsize=9,
            )

    plt.xticks(x_positions, ordered_models, rotation=45, ha="right")
    plt.xlabel("Model")
    plt.ylabel("Number of Errors (# Entities)")
    plt.title(title, fontsize=14, fontweight="semibold")
    plt.legend(title="Error Type", bbox_to_anchor=(1.02, 1), loc="upper left", title_fontsize="small")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def make_ner_few_shot_parallel_plot(data: pd.DataFrame, title: str, save_path: str= None, metrics: list[str] = ["f1 overall - strict", "precision overall - strict", "recall overall - strict"]):
    """ Plot few-shot NER performance across models and conditions with error bars."""
    sns.set_style("darkgrid")
    df = data.copy()
    df['model'] = df['model'].apply(lambda x: 'bert-baseline' if 'bert' in x.lower() else x)

    rename_map = {
        'zero_shot': '0', 
        'selected_1shot': '1', 
        'selected_3shot': '3', 
        'selected_5shot': '5',
    }
    df['condition'] = df['condition'].replace(rename_map)
    metric_rename_map = {'f1 overall - strict': 'F1 BIO - strict',
        'precision overall - strict': 'Precision BIO - strict',
        'recall overall - strict': 'Recall BIO - strict'}
    # rename columns according to rename_map
    df = df.rename(columns=metric_rename_map)
    metrics = [metric_rename_map[m] for m in metrics]
    
    condition_order = ['0', '1', '3', '5']


    bert_df = df[(df['model'] == 'bert-baseline') & (df['condition'] == '0')]
    bert_metrics = {metric: bert_df[metric].iloc[0] for metric in metrics}

    df = df[~df['model'].isin(['tuned', 'bert-baseline'])]
    
    # Establish model and condition order for plotting
    models = [m for m in model_order if m in df['model'].unique()]
    df['model'] = pd.Categorical(df['model'], categories=models, ordered=True)
    df['condition'] = pd.Categorical(df['condition'], categories=condition_order, ordered=True)
    df = df.sort_values(['model', 'condition'])

    # Layout: roughly square grid
    ncols = len(metrics)
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(5 * ncols, 4), sharey=False)
    axes = np.array(axes).flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        # Plot the main lines
        sns.lineplot(
            data=df,
            x='condition',
            y=metric,
            hue='model',
            marker='.',
            ax=ax,
            hue_order=models,
            palette=model_color_map,
            linewidth=1.5,
        )

        # Apply linestyles according to texture_map
        for line, model in zip(ax.get_lines(), models):
            if model in texture_map:
                line.set_linestyle('--')
            else:
                line.set_linestyle('-')

        # Add bert-baseline as a horizontal reference line        
        ax.axhline(
            y=bert_metrics[metric],
            color=model_color_map.get('bert-baseline', 'gray'),
            linestyle='--',
            linewidth=1.5,
            label='bert-baseline'
        )
        ax.set_ylim(0, 1)
        # set y label to metric
        ax.set_ylabel(metric)
        ax.set_xlabel("")
        ax.legend().remove()

    rank_marker_map = {1: '*', 2: '^', 3: 'v'}

    for i, metric in enumerate(metrics):
        ax = axes[i]
        rank_df = pd.concat([
            df[['model', 'condition', metric]],
            bert_df[['model', 'condition', metric]] if metric in bert_metrics else pd.DataFrame()
        ]).dropna(subset=[metric])

        top3 = rank_df.nlargest(3, metric)
        previous_pos = [(0, 0)]
        for rank, (_, row) in enumerate(top3.iterrows(), start=1):
            x = condition_order.index(row['condition'])
            y = row[metric]
            model = row['model']

            ax.scatter(
                x=x,
                y=y,
                s=100,
                color=model_color_map.get(model, 'black'),
                marker=rank_marker_map[rank],
                zorder=10,
                edgecolor=model_color_map.get(model, 'black')
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
                   color=model_color_map.get(model, 'gray'),
                   linestyle='--' if model in texture_map else '-',
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

    fig.suptitle(title, fontsize=16, fontweight='semibold',
                 color="#222222", y=0.98)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def make_ift_class_performance_plot(data: pd.DataFrame, best_model: pd.DataFrame):
    """Vertical bar plots per task in a fixed grid (4 columns x 5 rows).
    Each subplot: x = model (vertical bars), y = f1-weighted.
    Best-model condition is annotated inside the corresponding bar.
    """
    combined = pd.concat([
        data.assign(source="IFT/Baseline"),
        best_model.assign(source="Best")
    ], ignore_index=True)

    tasks = sorted(combined["task"].unique())
    n_tasks = len(tasks)

    # fixed grid 4 cols x 5 rows
    ncols = 5
    nrows = 4
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(5 * ncols, 4 * nrows),
                             sharey=False)
    axes = axes.flatten()

    default_palette = sns.color_palette("tab10")
    for i, task in enumerate(tasks):
        ax = axes[i]
        subset = combined[combined["task"] == task].copy()

        # determine an ordered model list that respects global model_order first
        unique_models = list(subset['model'].unique())

        # prepare palette for the models in this subset
        palette = [model_color_map.get(m, default_palette[idx % len(default_palette)])
                   for idx, m in enumerate(unique_models)]
        best_model_name = best_model[best_model['task'] == task]['model'].iloc[0] if not best_model[best_model['task'] == task].empty else None
        fixed_order = [
            'bert-baseline',
            'Llama-3.1-8B-Instruct-IFT',
            'Llama-3.1-8B-Instruct'
        ]
        plot_order = []
        if best_model_name:
            plot_order.append(best_model_name)
        # Add the rest, ensuring no duplicates and that they exist in the subset
        for model in fixed_order:
            if model != best_model_name and model in subset['model'].values:
                plot_order.append(model)
        


        # vertical barplot: x=model, y=f1-weighted
        sns.barplot(
            data=subset,
            x="model",
            y="f1-weighted",
            order=plot_order,
            palette=palette,
            ax=ax,
            errorbar=None,
        )

        # rotate x labels for readability
        ax.set_xticklabels(unique_models, rotation=45, ha="right", fontsize=9)

        # annotate numeric F1 values above each bar and optionally the condition inside best bars
        metric_by_model = subset.set_index('model')['f1-weighted'].to_dict()
        best_subset = best_model[best_model["task"] == task]

        # iterate over bars (patches) which correspond to ordered_models
        for idx, patch in enumerate(ax.patches):
            model = plot_order[idx]
            value = float(metric_by_model.get(model, 0.0))
            x_center = patch.get_x() + patch.get_width() / 2
            y_top = patch.get_height()

            # numeric label above bar
            ax.text(x_center, y_top + 0.01, f"{value:.2f}",
                    ha="center", va="bottom", fontsize=9, color="black")

        # annotate the condition for best models inside their bars (centered)
        for _, bm in best_subset.iterrows():
            mname = bm["model"]
            if mname in plot_order:
                idx_model = plot_order.index(mname)
                # try to get corresponding patch
                try:
                    patch = ax.patches[idx_model]
                except IndexError:
                    continue
                x_center = patch.get_x() + patch.get_width() / 2
                y_center = patch.get_height() / 2
                cond_text = str(bm.get("condition", "")).replace("selected_", "").replace("_", "-")
                # choose text color for contrast
                text_color = "white" if patch.get_height() > 0.5 else "black"
                ax.text(x_center, y_center, cond_text,
                        ha="center", va="center",
                        color=text_color, fontsize=9, fontweight="bold")

        ax.set_title(task.replace("_", " ").title(), fontsize=12)
        ax.set_xlabel("")
        # remove x axis ticker
        ax.set_xticks([])
        ax.set_ylabel("")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Remove any extra (unused) axes
    for j in range(n_tasks, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.subplots_adjust(top=0.94, bottom=0.06)
    fig.suptitle("IFT Class-wise Model Performance (with Best Models)", fontsize=16, fontweight="bold")

    # Legend for present models (preserve order as appeared in combined)
    present_models = []
    for m in model_order:
        if m in combined['model'].unique():
            present_models.append(m)
    # add any remaining models that are not in model_order
    present_models += [m for m in combined['model'].unique() if m not in present_models]

    handles = [plt.Rectangle((0, 0), 1, 1, color=model_color_map.get(m, '#cccccc')) for m in present_models]
    fig.legend(handles, present_models, title="Model", loc="lower center", ncol=min(6, len(present_models)))

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
