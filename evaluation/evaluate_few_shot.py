import os
import sys
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from plots.plots import make_performance_plot, make_few_shot_performance_plot
from evaluation.evaluate_zero_shot import (
    add_bert_performance_data,
    get_all_prediction_files,
    parse_file_name,
    evaluate_predictions,
    convert_numpy
)
from zero_shot.predict_zero_shot import parse_class_predictions, parse_ner_predictions
from prompts.build_prompts import get_label2int

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
    print(performance_data)
    # Make performance plot
    make_few_shot_performance_plot(performance_data)

def main():
    TASKS = [
        "Study Type",  
        # "Data Collection", "Data Type",
        # "Number of Participants", "Age of Participants", "Application Form",
        # "Clinical Trial Phase", "Condition", "Outcomes", "Regimen", "Setting", "Study Control", "Study Purpose",
        # "Substance Naivety", "Substances", "Sex of Participants", "Study Conclusion", "Relevant"
    ]
    for task in TASKS:
        parsing_log = os.path.join(
            PREDICTION_DIR, task.lower().replace(' ', '_'), 'parsing_log.txt')
        if os.path.exists(parsing_log):
            os.remove(parsing_log)
        with open(parsing_log, "a") as log_file:
            report_data = defaultdict(lambda: defaultdict(dict))

            print(f"Evaluating task: {task}")
            for pred_file in tqdm(get_all_prediction_files(PREDICTION_DIR, task)):
                print(f"\tParsing: {pred_file}")
                # 1. Parse class predictions
                parsing_report = parse_class_predictions(
                    pred_file, task, reparse=True, log_file=log_file)
                # 2. Evaluate predictions
                performance_report = evaluate_predictions(pred_file, task)
                model = parse_file_name(pred_file, "model")
                condition = parse_file_name(pred_file, "condition")
                report_data[model][condition].update(performance_report)
                report_data[model][condition].update(parsing_report)

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


    plot_performance_report('Study Type', 'few_shot/study_type/performance_report.json')

if __name__ == "__main__":
    main()