from plots.plots import make_performance_plot
from evaluation.evaluate import get_performance_report
from zero_shot.predict_zero_shot import parse_class_predictions, parse_ner_predictions
import numpy as np
import pandas as pd
import json
import sys
import os
from prompts.build_prompts import get_label2int


# enable import from parent directory
sys.path.append(os.path.abspath('..'))

PREDICTION_DIR = 'zero_shot'

performance_dicts = {}  # Skip non-CSV files


def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def evaluate_model_predictions(task: str, prediction_path: str) -> str:
    label2int = get_label2int(task)

    performance_dicts = {}
    for file in os.listdir(prediction_path):
        if not file.endswith('.csv'):
            continue
        print(f"Processing file: {file}")
        file_path = os.path.join(prediction_path, file)
        parsing_stats = parse_class_predictions(file_path, task)
        file_parts = file.split('_')
        model = file_parts[-2] + '_' + file_parts[-1].replace('.csv', '')
        df = pd.read_csv(file_path)
        performance_dict = {}
        performance_dict.update(parsing_stats)
        df_without_empty = df[df['pred_labels'].notna() & (
            df['pred_labels'] != '')]
        # Fill empty prediction with onehot encoding of all zeros
        df_with_empty = df.copy()
        df_with_empty['pred_labels'] = df_with_empty['pred_labels'].fillna(
             str([0]*len(label2int)))
        # Rename metrics to metrics_without_empty
        performance_dict['metrics_without_empty'] = get_performance_report(
            'labels', 'pred_labels', df_without_empty).pop('metrics')
        performance_dict.update(get_performance_report(
            'labels', 'pred_labels', df_with_empty))    
        performance_dicts[model] = performance_dict

    # Convert numpy types to native Python types
    performance_dicts = convert_numpy(performance_dicts)

    # dump the performance report
    performance_report_path = os.path.join(
        prediction_path, "performance_report.json")
    os.makedirs(prediction_path, exist_ok=True)
    with open(performance_report_path, "w") as f:
        json.dump(performance_dicts, f, indent=4)

    return performance_report_path


def add_bert_performance_data(task: str, performance_data_path: str):
    with open(performance_data_path, "r") as f:
        performance_data = json.load(f)

    bert_data = {}
    bert_path = '/home/vera/Documents/Uni/Master/Master_Thesis2.0/PsyNamic-Scale/bert_baseline'
    model = None
    for metric in ['f1', 'accuracy', 'precision', 'recall']:
        # find best_accuracy_scores.csv ect. files
        file_path = f'best_{metric}_scores.csv'
        df = pd.read_csv(os.path.join(bert_path, file_path))
        # get row from task
        task_row = df[df['task'] == task]

        score = float(task_row[f'{metric}_score'].values[0])
        ci_lower = float(task_row['ci_lower'].values[0])
        ci_upper = float(task_row['ci_upper'].values[0])
        score_data = [score, [ci_lower, ci_upper]]

        if metric == 'f1':
            metric = 'f1-weighted'
        bert_data[metric] = score_data
        # Assuming the model is the same for all metrics
        model = task_row['model'].values[0]

    # Add clinicalbert to performance_data if not present
    performance_data[model] = {'metrics': {},
                               'nr_faulty_parsable': 0, 'nr_non_parsable': 0}

    performance_data[model]['metrics'] = bert_data

    # save new performance data
    with open(performance_data_path, "w") as f:
        json.dump(performance_data, f, indent=4)

    return performance_data


def main():
    TASKS = [
    "Study Type", "Data Collection", "Data Type", "Number of Participants", "Age of Participants", "Application Form",
    "Clinical Trial Phase", "Condition", "Outcomes", "Regimen", "Setting", "Study Control", "Study Purpose",
    "Substance Naivety", "Substances", "Sex of Participants", "Study Conclusion", "Relevant"
]   
    for task in TASKS:
        prediction_path = os.path.join(
            PREDICTION_DIR, task.lower().replace(' ', '_'))
        performance_report_path = evaluate_model_predictions(task, prediction_path)
        performance_data = add_bert_performance_data(task, performance_report_path)
        plot_path = os.path.join(prediction_path, "performance_plot.png")
        make_performance_plot(performance_data, plot_path, metrics_col='metrics')
        make_performance_plot(performance_data, plot_path.replace(
            '.png', '_without_empty.png'), metrics_col='metrics_without_empty')


if __name__ == "__main__":
    main()
