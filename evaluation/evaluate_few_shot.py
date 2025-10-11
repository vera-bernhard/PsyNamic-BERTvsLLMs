import os
import sys
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from plots.plots import make_few_shot_performance_plot
from evaluation.evaluate_zero_shot import (
    add_bert_performance_data,
    get_all_prediction_files,
    parse_file_name,
    parse_all_class_predictions,
    evaluate_predictions,
    convert_numpy,
    TASKS,
)

# enable import from parent directory
sys.path.append(os.path.abspath('..'))

PREDICTION_DIR = 'few_shot'

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
            performance_data[model]['zero_shot']['metrics'] = zero_shot_data[model].get('metrics', {})
    # Make performance plot
    make_few_shot_performance_plot(performance_data, plot_path)

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
            if model in performance_reports and not reevaluate:
                print(
                    f"Skipping evaluation for {model} as it is already evaluated.")
                continue
            performance_report = evaluate_predictions(pred_file, task)
            performance_reports[model] = performance_reports.get(model, {})
            performance_reports[model][condition] = performance_report

        with open(performance_reports_path, "w") as f:
            json.dump(performance_reports, f, indent=4)

        # Step 3: Add BERT performance data
        # performance_reports = add_bert_performance_data(
        #     task, performance_reports_path)

        # Step 4: Make performance plot
        plot_path = os.path.join(
            prediction_dir, task.lower().replace(' ', '_'), "performance_plot.png")
        make_few_shot_performance_plot(performance_reports, plot_path,
                              metric='f1-weighted')


def main():
    PREDICTION_DIR = 'few_shot'
    parse_all_class_predictions(TASKS, PREDICTION_DIR, reparse=True)
    evaluate_all_class_tasks(TASKS, PREDICTION_DIR, reevaluate=True)
    # f = 'few_shot/number_of_participants/number_of_participants_1shot_selected_Llama-2-13b-chat-hf_27-09-27.csv'
    # task = 'Number of Participants'
    # evaluate_predictions(f, task)

    # parse_all_ner_predictions(PREDICTION_DIR)
    # evaluate_all_ner(PREDICTION_DIR)


    # for task in TASKS:
    #     parsing_log = os.path.join(
    #         PREDICTION_DIR, task.lower().replace(' ', '_'), 'parsing_log.txt')
    #     if os.path.exists(parsing_log):
    #         os.remove(parsing_log)
    #     with open(parsing_log, "a") as log_file:
    #         report_data = defaultdict(lambda: defaultdict(dict))

    #         print(f"Evaluating task: {task}")
    #         for pred_file in tqdm(get_all_prediction_files(PREDICTION_DIR, task)):
    #             print(f"\tParsing: {pred_file}")
    #             # 1. Parse class predictions
    #             parsing_report = parse_class_predictions(
    #                 pred_file, task, reparse=True, log_file=log_file)
    #             # 2. Evaluate predictions
    #             performance_report = evaluate_predictions(pred_file, task)
    #             model = parse_file_name(pred_file, "model")
    #             condition = parse_file_name(pred_file, "condition")
    #             report_data[model][condition].update(performance_report)
    #             report_data[model][condition].update(parsing_report)

    #     # 3. Save performance report of the task
    #     performance_report_path = os.path.join(
    #         PREDICTION_DIR, task.lower().replace(' ', '_'), "performance_report.json")
    #     # Convert numpy types to native Python types
    #     report_data = convert_numpy(report_data)
    #     with open(performance_report_path, "w") as f:
    #         json.dump(report_data, f, indent=4)

        # # 4. Add BERT performance data
        # report_data = add_bert_performance_data(task, performance_report_path)
        # plot_path = os.path.join(
        #     PREDICTION_DIR, task.lower().replace(' ', '_'), "performance_plot.png")
        
    # for task in TASKS:
    #     performance_report_path = os.path.join(
    #         PREDICTION_DIR, task.lower().replace(' ', '_'), "performance_report.json")
    #     if os.path.exists(performance_report_path):
    #         plot_performance_report(task, performance_report_path)


if __name__ == "__main__":
    main()