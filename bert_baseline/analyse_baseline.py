import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from evaluation.evaluate import evaluate_ner_extraction, evaluate_ner_bio, bootstrap_metrics, ner_error_analysis
from evaluation.evaluate_zero_shot import get_ner_predictions_and_labels


TASKS = [
    "Data Collection", "Data Type", "Number of Participants", "Age of Participants", "Application Form",
    "Clinical Trial Phase", "Condition", "Outcomes", "Regimen", "Setting", "Study Control", "Study Purpose",
    "Substance Naivety", "Substances", "Sex of Participants", "Study Conclusion", "Study Type", "Relevant", 
    "NER"
]

MODEL_COLORS = {
    "pubmedbert": "#1f77b4", "biomedbert-abstract": "#ff7f0e", "scibert": "#2ca02c",
    "biobert": "#d62728", "clinicalbert": "#9467bd", "biolinkbert": "#8c564b"
}


# Written for MA
def plot_best_class_scores(directory: str, metric: str = 'F1', best_strategy: str = 'F1') -> None:
    df_tasks = get_best_class_scores(directory, metric, best_strategy)
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar plot without built-in error bars
    sns.barplot(
        x="task",
        y="f1_score",
        hue="model",
        data=df_tasks,
        palette=MODEL_COLORS,
        ax=ax,
        legend=False,
        errorbar=None
    )

    # Add error bars manually
    for index, row in df_tasks.iterrows():
        yerr_lower = row["f1_score"] - row["ci_lower"]
        yerr_upper = row["ci_upper"] - row["f1_score"]
        ax.errorbar(index, row["f1_score"],
                    yerr=[[yerr_lower], [yerr_upper]],
                    fmt='none', color='black', capsize=5)

        # Display values
        ax.text(index, row["f1_score"] - 0.2,
                f"{row['f1_score']:.3f}", ha="center", color="black", fontsize=10)

        ax.text(index, row["ci_lower"] - 0.04,
                f"{row['ci_lower']:.3f}", ha="center", va="bottom", color="black", fontsize=8)

        ax.text(index, row["ci_upper"] + 0.01,
                f"{row['ci_upper']:.3f}", ha="center", va="bottom", color="black", fontsize=8)

    # Legend for colors
    for model, color in MODEL_COLORS.items():
        ax.bar(0, 0, color=color, label=model)
    ax.legend(title="Model", loc="upper left",
              bbox_to_anchor=(1, 1), title_fontsize="small")

    # Add some padding at bottom so labels are not cut off
    plt.gcf().subplots_adjust(bottom=0.2)

    # Formatting
    ax.set_ylabel(f"Best {metric} Score")
    ax.set_title(f"Best {metric} Score per Task")
    ax.set_xticks(np.arange(len(df_tasks)))
    ax.set_xticklabels(df_tasks["task"], rotation=45, ha="right")
    ax.set_ylim(0, 1)

    plt.show()


# Written for MA
def get_best_class_scores(directory: str, metric: str = 'F1', best_strategy: str = 'F1') -> pd.DataFrame:
    task_data = []

    # Iterate through CSV files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            task_name = filename.replace("model_performance_", "").replace(
                ".csv", "").replace("_", " ").title()

            # Load CSV
            df = pd.read_csv(os.path.join(directory, filename))

            if best_strategy not in df.columns and best_strategy != 'cur':
                raise ValueError(
                    "Available strategies: 'F1, 'Accuracy', 'Precision', 'Recall' or 'cur' (for current metric)")
            elif best_strategy == 'cur':
                best_strategy = metric
            best_model = df.loc[df[best_strategy].idxmax()]

            # Store relevant data
            task_data.append({
                "task": task_name,
                "model": best_model["Model"],
                f"{metric.lower()}_score": best_model[f"{metric}"],
                "ci_lower": best_model[f"{metric} CI Lower"],
                "ci_upper": best_model[f"{metric} CI Upper"]
            })

    # Convert to DataFrame
    df_tasks = pd.DataFrame(task_data)
    # Sort by F1 score for better visualization
    df_tasks = df_tasks.sort_values(
        by=f"{metric.lower()}_score", ascending=False).reset_index(drop=True)
    return df_tasks


def collect_ner_metrics(prediction_dir: str) -> None:
    model_performance = []
    model_error_analysis = {}
    for filename in os.listdir(prediction_dir):
        if filename.endswith("formatted.csv"):
            model = filename.split("_")[0]
            file_path = os.path.join(prediction_dir, filename)
            print(f"Evaluating {file_path}...")
            # Evaluate BIO
            # TODO: very ugly to have to call this function here again, s. evaluate_zero_shot.py
            pred_bio, pred_entities, true_bio, true_entities = get_ner_predictions_and_labels(file_path)
            r = bootstrap_metrics(evaluate_ner_extraction,
                              pred_entities, true_entities)
            r_bio = bootstrap_metrics(evaluate_ner_bio, pred_bio, true_bio)
            e = ner_error_analysis(pred_bio, true_bio)
            # merge two dicts
            r.update(r_bio)
            for metric in r:
                model_performance.append({
                    "Model": model,
                    "Metric": metric,
                    "Score": r[metric]['mean'],
                    "CI Lower": r[metric]['lower'],
                    "CI Upper": r[metric]['upper'],
                })
            model_error_analysis[model] = e
    df = pd.DataFrame(model_performance)
    # write into parents directory
    df.to_csv(os.path.join(os.path.dirname(prediction_dir), "ner_performance.csv"), index=False)
    with open(os.path.join(os.path.dirname(prediction_dir), "ner_error_analysis.json"), "w") as f:
        json.dump(model_error_analysis, f, indent=4)


def main():
    dir = 'model_performance'
    get_best_class_scores(dir, metric='F1').to_csv(
        os.path.join(os.path.dirname(dir), 'best_f1_scores_class.csv'), index=False)
    get_best_class_scores(dir, metric='Accuracy').to_csv(
        os.path.join(os.path.dirname(dir), 'best_accuracy_scores_class.csv'), index=False)
    get_best_class_scores(dir, metric='Precision').to_csv(
        os.path.join(os.path.dirname(dir), 'best_precision_scores_class.csv'), index=False)
    get_best_class_scores(dir, metric='Recall').to_csv(
        os.path.join(os.path.dirname(dir), 'best_recall_scores_class.csv'), index=False)

    collect_ner_metrics('bert_baseline/predictions')    

if __name__ == "__main__":
    main()
