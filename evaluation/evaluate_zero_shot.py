from plots.plots import make_performance_plot, make_simple_performance_plot
from evaluation.evaluate import get_performance_report
from zero_shot.predict_zero_shot import parse_class_predictions, parse_ner_predictions
import numpy as np
import pandas as pd
import json
import sys
import os
from prompts.build_prompts import get_label2int
from collections import defaultdict
from typing import Literal
from tqdm import tqdm


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


def replace_nan_with_empty_onehot(df: pd.DataFrame, label_col: str, label_int: int) -> pd.DataFrame:
    df = df.copy()
    num_labels = len(label_int)
    df[label_col] = df[label_col].apply(lambda x: str(
        [0]*num_labels) if pd.isna(x) or x == '' else x)
    return df


def remove_empty_predictions(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    return df[df[pred_col].notna() & (df[pred_col] != '')]


def evaluate_predictions(file_path: str, task: str, remove_empty: bool = False) -> dict:
    label2int = get_label2int(task)
    df = pd.read_csv(file_path)
    performance_dict = {}
    if remove_empty:
        df_without_empty = remove_empty_predictions(df, 'pred_labels')
        performance_dict['metrics_without_empty'] = get_performance_report(
            'labels', 'pred_labels', df_without_empty).pop('metrics')
    else:
        df_with_empty = replace_nan_with_empty_onehot(
            df, 'pred_labels', label2int)
        performance_report = get_performance_report(
            'labels', 'pred_labels', df_with_empty)
        # check if there were empty predictions
        if performance_report.get('nr_empty_predictions', 0) > 0:
            raise ValueError(
                f"There are {performance_report['nr_empty_predictions']} empty predictions in file {file_path}. Please fix the input data.")
        performance_dict.update(performance_report)
    return performance_dict


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
        # make task case insensitive
        df['task'] = df['task'].str.lower()
        task = task.lower()
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


def overall_class_performance(tasks: list[str], prediction_dir: str, metric: str = 'f1-weighted'):
    overall_performance = defaultdict(list)
    performance_reports = []
    # collect all performance reports
    for task in tasks:
        prediction_path = os.path.join(
            prediction_dir, task.lower().replace(' ', '_'), 'performance_report.json')
        # check if path exists
        if os.path.exists(prediction_path):
            performance_reports.append(prediction_path)
    # aggregate the performance reports
    for report_path in performance_reports:
        with open(report_path, "r") as f:
            performance_data = json.load(f)
        for model, data in performance_data.items():
            # get model name without prediction date
            model = model.split('_')[0]
            if 'bert' in model.lower():
                overall_performance['bert-baseline'].append(
                    data['metrics'][metric][0])
            else:
                overall_performance[model].append(data['metrics'][metric][0])
    # Check if all models have predicted all tasks
    num_tasks = len(performance_reports)
    for model, scores in overall_performance.items():
        if len(scores) != num_tasks:
            print(
                f"Warning: Model {model} has predictions for {len(scores)} out of {num_tasks} tasks.")
    # average the performance
    averaged_performance = {}
    for model, scores in overall_performance.items():
        averaged_performance[model] = np.mean(scores)
    return averaged_performance


def get_all_prediction_files(prediction_dir: str, task: str) -> list[str]:
    task_dir = os.path.join(
        prediction_dir, task.lower().replace(' ', '_'))
    return [os.path.join(task_dir, file) for file in os.listdir(task_dir) if file.endswith('.csv')]


def parse_file_name(pred_path: str, info: Literal["model", "condition", "date", "task"]) -> str:
    """Parses the prediction file name to extract model, condition, date or task.
    Example file names: 
    - setting_MeLLaMA-13B-chat_06-09-06.csv
    - clinical_trial_phase_1shot_selected_gpt-4o-2024-08-06_09-09-09.csv
    - sex_of_participants_gpt-4o-mini_09-09-09.csv
    """
    filename = os.path.basename(pred_path).replace('.csv', '')
    parts = filename.rsplit('_')

    if 'shot' in filename:
        date = parts[-1]
        model = parts[-2]
        condition = parts[-3]
        if info == "date":
            return date
        elif info == "model":
            return model
        elif info == "condition":
            return condition
        elif info == "task":
            return '_'.join(parts[:-3])
        else:
            raise ValueError(f"Unknown info type: {info}")
    else:
        date = parts[-1]
        model = parts[-2]
        if info == "date":
            return date
        elif info == "model":
            return model
        elif info == "task":
            return '_'.join(parts[:-2])
        else:
            raise ValueError(f"Unknown info type: {info}")

    raise ValueError(f"Filename does not match expected patterns: {filename}")


def main():
    TASKS = [
        # "Study Type",  
        "Data Collection", "Data Type",
        "Number of Participants", "Age of Participants", "Application Form",
        "Clinical Trial Phase", "Condition", "Outcomes", "Regimen", "Setting", "Study Control", "Study Purpose",
        "Substance Naivety", "Substances", "Sex of Participants", "Study Conclusion", "Relevant"
    ]
    for task in TASKS:
        parsing_log = os.path.join(
            PREDICTION_DIR, task.lower().replace(' ', '_'), 'parsing_log.txt')
        if os.path.exists(parsing_log):
            os.remove(parsing_log)
        with open(parsing_log, "a") as log_file:
            report_data = defaultdict(dict)
            print(f"Evaluating task: {task}")
            for pred_file in tqdm(get_all_prediction_files(PREDICTION_DIR, task)):
                print(f"\tParsing: {pred_file}")
                # 1. Parse class predictions
                parsing_report = parse_class_predictions(
                    pred_file, task, reparse=True, log_file=log_file)
                # 2. Evaluate predictions
                performance_report = evaluate_predictions(pred_file, task)
                model = parse_file_name(pred_file, "model")
                report_data[model].update(performance_report)
                report_data[model].update(parsing_report)

        # 3. Save performance report of the task
        performance_report_path = os.path.join(
            PREDICTION_DIR, task.lower().replace(' ', '_'), "performance_report.json")
        # Convert numpy types to native Python types
        report_data = convert_numpy(report_data)
        with open(performance_report_path, "w") as f:
            json.dump(report_data, f, indent=4)

        # 4. Add BERT performance data
        report_data = add_bert_performance_data(task, performance_report_path)
        plot_path = os.path.join(
            PREDICTION_DIR, task.lower().replace(' ', '_'), "performance_plot.png")
        # 5. Make performance plot
        make_performance_plot(report_data, plot_path,
                              metrics_col='metrics')

    # averaged_performance = overall_class_performance(TASKS, PREDICTION_DIR)
    # make_simple_performance_plot(averaged_performance, 'zero_shot/overall_performance.png')


if __name__ == "__main__":
    main()
