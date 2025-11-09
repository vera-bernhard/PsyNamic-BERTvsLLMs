import os
import sys
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from evaluation.evaluate import bootstrap_metrics, evaluate_ner_bio, evaluate_ner_extraction, ner_error_analysis
from plots.plots import make_few_shot_performance_plot, make_few_shot_box_plot, make_few_shot_trend_plot, make_few_shot_delta_plot, make_few_shot_avg_plot, make_few_shot_parallel_plot, make_ner_few_shot_parallel_plot, make_ner_few_shot_error_analysis_plot
from evaluation.evaluate_zero_shot import (
    add_bert_performance_data,
    get_all_prediction_files,
    parse_file_name,
    parse_all_class_predictions,
    parse_all_ner_predictions,
    evaluate_predictions,
    convert_numpy,
    TASKS,
    get_ner_predictions_and_labels
)

# enable import from parent directory
sys.path.append(os.path.abspath('..'))

PREDICTION_DIR = 'few_shot'
ZERO_SHOT_DIR = 'zero_shot'


def overall_class_performance(tasks: list[str], prediction_dir: str) -> pd.DataFrame:
    all_data = []
    for task in tasks:
        performance_report_path = os.path.join(
            prediction_dir, task.lower().replace(' ', '_'), 'performance_report.json'
        )
        with open(performance_report_path, "r") as f:
            performance_report = json.load(f)

        for model in performance_report:
            for condition in performance_report[model]:
                metrics = performance_report[model][condition]['metrics']
                for metric_name, metric_values in metrics.items():
                    row = {
                        'task': task,
                        'model': model,
                        'condition': condition,
                        'metric': metric_name,
                        'mean': metric_values[0],
                        'lower': metric_values[1][0],
                        'upper': metric_values[1][1],
                    }
        
                    all_data.append(row)
    # add zero-shot bert performance data
    for task in tasks:
        task_data = get_zero_shot_performance_data(task)
        for model in task_data:
            metrics = task_data[model]['metrics']
            for metric_name, metric_values in metrics.items():
                row = {
                    'task': task,
                    'model': model,
                    'condition': 'zero_shot',
                    'metric': metric_name,
                    'mean': metric_values[0],
                    'lower': metric_values[1][0],
                    'upper': metric_values[1][1],
                }
                all_data.append(row)

    df = pd.DataFrame(all_data)

    return df

def overall_ner_performance(prediction_dir: str) -> pd.DataFrame:
    performance_report_path = os.path.join(
        prediction_dir, 'ner', 'ner_performance_report.json'
    )
    all_data = []
    error_data = []
    with open(performance_report_path, "r") as f:
        performance_report = json.load(f)

    for model, conditions in performance_report.items():
        for condition, metrics in conditions.items():
            # Loop over metrics for this model-condition pair
            for metric_name, metric_values in metrics.items():
                if metric_name.lower() != 'errors':
                    all_data.append({
                            'model': model,
                            'condition': condition,
                            'metric': metric_name,
                            'mean': metric_values.get('mean'),
                            'lower': metric_values.get('lower'),
                            'upper': metric_values.get('upper')
                        })
                else:
                    # Handle errors
                    for error_type, error_count in metric_values.items():
                        error_data.append({
                            'model': model,
                            'condition': condition,
                            'error_type': error_type,
                            'count': error_count
                        })
    df = pd.DataFrame(all_data)
    error_df = pd.DataFrame(error_data)
    return df, error_df

def get_zero_shot_performance_data(task: str):
    zero_shot_report_path = os.path.join(
        ZERO_SHOT_DIR, task.lower().replace(' ', '_'), "performance_reports.json")
    with open(zero_shot_report_path, "r") as f:
        zero_shot_data = json.load(f)
    return zero_shot_data


def plot_performance_report(task: str, performance_report: str):
    # Load performance report
    with open(performance_report, "r") as f:
        performance_data = json.load(f)
    plot_path = os.path.join(
        PREDICTION_DIR, task.lower().replace(' ', '_'), "performance_plot.png")

    # Add the zero_shot performance data for comparison
    zero_shot_report_path = os.path.join(
        'zero_shot', task.lower().replace(' ', '_'), "performance_report.json")
    with open(zero_shot_report_path, "r") as f:
        zero_shot_data = json.load(f)

    for model in zero_shot_data:
        if model in performance_data:
            performance_data[model]['zero_shot'] = {}
            performance_data[model]['zero_shot']['metrics'] = zero_shot_data[model].get(
                'metrics', {})
    # Make performance plot
    make_few_shot_performance_plot(performance_data, task,  plot_path)


def evaluate_all_class_tasks(tasks: list[str], prediction_dir: str, reevaluate: bool = False):
    for task in tasks:
        all_pred_files = get_all_prediction_files(prediction_dir, task)
        performance_reports_path = os.path.join(
            prediction_dir, task.lower().replace(' ', '_'), 'performance_report.json'
        )

        if os.path.exists(performance_reports_path) and not reevaluate:
            with open(performance_reports_path, "r") as f:
                performance_reports = json.load(f)
        else:
            performance_reports = {}

        for pred_file in tqdm(all_pred_files):
            print(f"Evaluating: {pred_file}")
            model = parse_file_name(pred_file, "model")
            condition = parse_file_name(pred_file, "condition")
            if model in performance_reports and not reevaluate and condition in performance_reports[model]:
                print(
                    f"Skipping evaluation for {model} as it is already evaluated.")
                continue
            performance_report = evaluate_predictions(pred_file, task)
            performance_reports[model] = performance_reports.get(model, {})
            performance_reports[model][condition] = performance_report

        with open(performance_reports_path, "w") as f:
            json.dump(performance_reports, f, indent=4)

        # Add zero-shot performance data for comparison
        zero_shot_data = get_zero_shot_performance_data(task)

        for model in zero_shot_data:
            if model in performance_reports:
                performance_reports[model]['zero_shot'] = {}
                performance_reports[model]['zero_shot']['metrics'] = zero_shot_data[model].get(
                    'metrics', {})

        # Step 4: Make performance plot
        plot_path = os.path.join(
            prediction_dir, task.lower().replace(' ', '_'), "performance_plot.png")
        make_few_shot_performance_plot(performance_reports, task, plot_path,
                                       metric='f1-weighted')


def evaluate_all_ner_tasks(prediction_dir: str, reevaluate: bool = False):

    all_pred_files = get_all_prediction_files(prediction_dir, task='ner')
    performance_reports_path = os.path.join(
        prediction_dir, 'ner', 'ner_performance_report.json'
    )

    if os.path.exists(performance_reports_path) and not reevaluate:
        with open(performance_reports_path, "r") as f:
            performance_reports = json.load(f)
    else:
        performance_reports = {}

    for pred_file in tqdm(all_pred_files):
        print(f"Evaluating NER: {pred_file}")
        model = parse_file_name(pred_file, "model")
        condition = parse_file_name(pred_file, "condition")
        if model in performance_reports and not reevaluate and condition in performance_reports[model]:
            print(
                f"Skipping NER evaluation for {model} as it is already evaluated.")
            continue

        pred_bio, pred_entities, true_bio, true_entities = get_ner_predictions_and_labels(pred_file)

        r = bootstrap_metrics(evaluate_ner_extraction,
                            pred_entities, true_entities)
        r_bio = bootstrap_metrics(evaluate_ner_bio, pred_bio, true_bio)
        error_analysis = ner_error_analysis(pred_bio, true_bio)
        performance_reports[model] = performance_reports.get(model, {})
        performance_reports[model][condition] = {**r, **r_bio}
        performance_reports[model][condition]['errors'] = error_analysis

    # Add zero-shot performance data for comparison
    zero_shot_data = get_zero_shot_performance_data('ner')
    for model in zero_shot_data:
        if model not in performance_reports:
            performance_reports[model] = {}
        performance_reports[model]['zero_shot'] = zero_shot_data.get(model, {})
        performance_reports[model]['zero_shot']['errors'] = zero_shot_data[model].get('errors', {})

    with open(performance_reports_path, "w") as f:
        json.dump(performance_reports, f, indent=4)
        
    
    make_few_shot_performance_plot(performance_reports, 'Named Entity Recognition', os.path.join(
        prediction_dir, 'ner', "ner_performance_plot.png"), metric='f1 overall - strict')


def main():
    PREDICTION_DIR = 'few_shot'
    # parse_all_class_predictions(TASKS, PREDICTION_DIR, reparse=False)
    # evaluate_all_class_tasks(TASKS, PREDICTION_DIR, reevaluate=False)

    # parse_all_ner_predictions(PREDICTION_DIR, reparse=False)
    # evaluate_all_ner_tasks(PREDICTION_DIR, reevaluate=False)
    all_ner, all_error = overall_ner_performance(PREDICTION_DIR)
    make_ner_few_shot_parallel_plot(all_ner, 'In-context Learning Performance across Models and Conditions', 'few_shot/overall_ner_parallel_plot.png')
    all_ner['task'] = 'NER'
    # make_few_shot_delta_plot(all_ner, 'Relative Improvement Δ F1 BIO - strict for 1-, 3-, 5-shot over Zero-Shot for NER',
    #                          'few_shot/overall_ner_relative_delta_plot.png', metric = 'f1 overall - strict')
    # make_ner_few_shot_error_analysis_plot(all_error, title='In-context Learning NER - Error Analysis',
    #                              save_path='few_shot/overall_ner_error_analysis.png')
    # Make parallel plot for subtasks
    all_ner_filtered = all_ner[all_ner['metric'].str.startswith('f1') & all_ner['metric'].str.endswith('strict')]
    all_ner_filtered['task'] = all_ner_filtered['metric'].apply(lambda x: 'Overall' if 'overall' in x else ('Dosage' if 'DOS' in x else 'Application'))
    all_ner_filtered['metric'] = 'F1 BIO - strict'
    make_few_shot_parallel_plot(
         all_ner_filtered, 'In-context Learning Performance across Models and Conditions for Named Entity Types', 'few_shot/overall_ner_subtasks_parallel_plot.png', metric='F1 BIO - strict')

    all_class = overall_class_performance(TASKS, PREDICTION_DIR)
    # make_few_shot_box_plot(all_class, 'few_shot/overall_class_performance_box_plot.png',
    # #                      metric='f1-weighted')
    # make_few_shot_trend_plot(all_class, 'F1 Score for 0-, 1-, 3-, 5-shot over All Classification Tasks',
    #                          'few_shot/overall_class_trend_plot_sharedy.png')
    # make_few_shot_trend_plot(all_class, 'F1 Score for 0-, 1-, 3-, 5-shot over All Classification Tasks',
    #                          'few_shot/overall_class_trend_plot.png', sharey=False)
    # make_few_shot_delta_plot(all_class, 'Relative Improvement Δ F1 for 1-, 3-, 5-shot over Zero-Shot for All Classification Tasks',
    #                          'few_shot/overall_class_relative_delta_plot.png')
    # make_few_shot_avg_plot(all_class, 'Average Performance over All Classification Tasks',
    #                         'few_shot/overall_class_average_performance_plot.png')
    make_few_shot_parallel_plot(
         all_class, 'In-context Learning Performance across Models and Conditions for All Classification Tasks', 'few_shot/overall_class_parallel_plot.png')


if __name__ == "__main__":
    main()
