import sys
import os

# enable import from parent directory
sys.path.append(os.path.abspath('..'))

import os
import json
import pandas as pd
import numpy as np
from zero_shot.predict_zero_shot import parse_class_predictions, parse_ner_predictions
from evaluation.evaluate import get_performance_report
from nervaluate import Evaluator, summary_report_ent, summary_report_overall
import pandas as pd
import ast
# plot results for all models, one plot per metric using seaborn
import seaborn as sns
import matplotlib.pyplot as plt

# for file in os.listdir('zero_shot/ner'):
#     if not file.endswith('.csv'):
#         continue
#     print(f"Processing file: {file}")
#     file_path = os.path.join('zero_shot/ner', file)
#     parse_ner_predictions(file_path, 'entities')
#     file_parts = file.split('_')
#     model = file_parts[-2] + '_' + file_parts[-1].replace('.csv', '')
#     print(model)
#     df = pd.read_csv(file_path)

# Add gold labels to the DataFrame
# get predictions and labels from csv files
def get_predictions_and_labels(file, pred_col, true_col):
    df = pd.read_csv(file)
    predictions = [ast.literal_eval(item) for item in df[pred_col]]
    labels = [ast.literal_eval(item) for item in df[true_col]]
    return predictions, labels


def get_performance(file:str, pred_col='pred_labels', true_col='ner_tags'):
    true, pred = get_predictions_and_labels(file, pred_col=pred_col, true_col=true_col)
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

model_performance = {}
for file in os.listdir('zero_shot/ner'):
    if not file.endswith('.csv'):
        continue
    print(f"Processing file: {file}")
    model = file.split('_')[-2] + '_' + file.split('_')[-1].replace('.csv', '')
    file_path = os.path.join('zero_shot/ner', file)
    model_performance[model] = get_performance(file_path)

print(model_performance)

model_performance['biomedbert-abstract_ner_bio'] = {'f1 overall - strict': 0.7251184834123223, 'f1 overall - partial': 0.7843601895734597, 'f1 APP - strict': 0.7067137809187279, 'f1 APP - partial': 0.7597173144876326, 'f1 DOS - strict': 0.762589928057554, 'f1 DOS - partial': 0.8345323741007195}

df = pd.DataFrame(model_performance).T.reset_index().rename(columns={"index": "Model"})
df_melted = df.melt(id_vars="Model", var_name="Metric", value_name="Score")
print(df_melted["Model"].unique())

# Set the plotting style
sns.set(style="whitegrid")

plt.figure(figsize=(14, 6))
barplot = sns.barplot(data=df_melted, x="Metric", y="Score", hue="Model")

# Annotate each bar with the score
for container in barplot.containers:
    barplot.bar_label(container, fmt="%.2f", label_type="edge", padding=3)

# Adjust the plot
plt.ylim(0, 1)
plt.title("F1 Score Comparison by Metric and Model")
plt.ylabel("F1 Score")
plt.xticks(rotation=45)

# Call legend AFTER tight_layout
plt.subplots_adjust(right=0.8)
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()