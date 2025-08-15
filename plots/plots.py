import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from evaluation.evaluate import get_performance_report

BERT_PRED = '/home/vera/Documents/Uni/Master/Master_Thesis2.0/PsyNamic-Scale/bert_baseline/predictions'


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
    MODEL_COLORS = dict(zip(unique_models, palette))

    sns.set(style="whitegrid")
    metrics = df_metrics['metric'].unique()

    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 8), sharey=True)

    if len(metrics) == 1:
        axes = [axes]  # ensure iterable if only one metric

    for i, metric in enumerate(metrics):
        ax = axes[i]
        df_sub = df_metrics[df_metrics['metric'] == metric].reset_index(drop=True)
        
        sns.barplot(
        x="model",
        y="mean",
        hue="model",
        data=df_sub,
        palette=MODEL_COLORS,
        ax=ax,
        errorbar=None,
        legend=False
    )

        ax.set(ylim=(0,1))

        for idx, row in df_sub.iterrows():
            yerr_lower = row["mean"] - row["ci_lower"]
            yerr_upper = row["ci_upper"] - row["mean"]
            ax.errorbar(idx, row["mean"],
                        yerr=[[yerr_lower], [yerr_upper]],
                        fmt='none', color='black', capsize=5)
            ax.text(idx, 0.02, f"{row['mean']:.3f}", ha="center", va="bottom", color="black", fontsize=10)
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
    handles = [plt.Rectangle((0,0),1,1, color=color) for model, color in MODEL_COLORS.items()]
    labels = list(MODEL_COLORS.keys())
    fig.legend(handles, labels, title="Model", loc="upper left", bbox_to_anchor=(0.8, 0.9), title_fontsize="small")

    plt.subplots_adjust(right=0.8, bottom=0.2)
    plt.show()


