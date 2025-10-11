from plots.plots import make_performance_plot, make_performance_box_plot, make_performance_spider_plot, make_simple_performance_plot
from evaluation.evaluate import get_performance_report, evaluate_ner_bio, evaluate_ner_extraction, bootstrap_metrics
from zero_shot.predict_zero_shot import parse_class_predictions, parse_ner_predictions
import numpy as np
import pandas as pd
import json
import sys
import os
from prompts.build_prompts import get_label2int
from tqdm import tqdm
from evaluation.parsing import parse_file_name, add_entities
import ast


# enable import from parent directory
sys.path.append(os.path.abspath('..'))

TASKS = [
        "Study Type", "Data Collection", "Data Type", "Number of Participants", "Age of Participants", "Application Form", "Clinical Trial Phase", "Condition", "Outcomes", "Regimen", "Study Control", "Study Purpose",
        "Substance Naivety", "Substances", "Study Conclusion", "Relevant",
        "Setting", "Sex of Participants",
    ]

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
    if 'ner' in task.lower():
        # read ner_performance.csv
        file_path = 'ner_performance.csv'
        df = pd.read_csv(os.path.join(bert_path, file_path))
        
        # get model with highest f1 overall - strict, metric == 'f1 overall - strict'
        best_row_overall_strict = df[df['Metric'] == 'f1 overall - strict'].sort_values(by='Score', ascending=False).iloc[0]
        best_row_entity_type = df[df['Metric'] == 'f1_entity_type'].sort_values(by='Score', ascending=False).iloc[0]
        # Check if it is the same model
        if best_row_overall_strict['Model'] != best_row_entity_type['Model']:
            print("Warning: Different models have the best scores for 'f1 overall - strict' and 'f1_entity_type'. Using the model with the best 'f1 overall - strict'.")
        model = best_row_overall_strict['Model']

        # add all metrics to the performance data
        # collect all unique
        all_metrics = df['Metric'].unique()
        performance_data[model] = {}
        # add to performance data
        for metric in all_metrics:
            # mean, lower, upper
            metric_row = df[(df['Metric'] == metric) & (df['Model'] == model)]
            mean = float(metric_row['Score'].values[0])
            ci_lower = float(metric_row['CI Lower'].values[0])
            ci_upper = float(metric_row['CI Upper'].values[0])
            score_data = {
                "mean": mean,
                "lower": ci_lower,
                "upper": ci_upper
            }
            performance_data[model][metric] = score_data

    else:
        for metric in ['f1', 'accuracy', 'precision', 'recall']:
            # find best_accuracy_scores.csv ect. files
            file_path = f'best_{metric}_scores_class.csv'
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


def overall_class_performance(tasks: list[str], prediction_dir: str, metric: str = 'f1-weighted') -> pd.DataFrame:
    rows = []
    for task in tasks:
        prediction_path = os.path.join(
            prediction_dir, task.lower().replace(' ', '_'), 'performance_reports.json')
        if not os.path.exists(prediction_path):
            continue
        with open(prediction_path, "r") as f:
            performance_data = json.load(f)
        for model, data in performance_data.items():
            model_name = model.split('_')[0]
            if 'bert' in model_name.lower():
                model_name = 'bert-baseline'
            score = data['metrics'].get(metric, [None])[0]
            rows.append(
                {'model': model_name, 'task': task, 'performance': score})
    df = pd.DataFrame(rows)
    return df


def overall_ner_performance(prediction_dir: str, metrics: list[str] = None) -> pd.DataFrame:
    rows = []
    prediction_path = os.path.join(
        prediction_dir, 'ner', 'performance_reports.json')
    if not os.path.exists(prediction_path):
        return pd.DataFrame(rows)
    with open(prediction_path, "r") as f:
        performance_data = json.load(f)
    for model, data in performance_data.items():
        model_name = model.split('_')[0]
        if 'bert' in model_name.lower():
            model_name = 'bert-baseline'
        for metric in metrics:
            score = data.get(metric, None)
            rows.append(
                {
                    'metric': metric,
                    'model': model_name,
                    'mean': score['mean'],
                    'ci_lower': score['lower'],
                    'ci_upper': score['upper']
                })
    df = pd.DataFrame(rows)
    return df


def get_all_prediction_files(prediction_dir: str, task: str) -> list[str]:
    task_dir = os.path.join(
        prediction_dir, task.lower().replace(' ', '_'))
    return [os.path.join(task_dir, file) for file in os.listdir(task_dir) if file.endswith('.csv')]


def plot_performance_report(task: str, performance_report: str):
    # Load performance report
    with open(performance_report, "r") as f:
        performance_data = json.load(f)
    plot_path = os.path.join(
        PREDICTION_DIR, task.lower().replace(' ', '_'), "performance_plot.png")
    # Make performance plot
    make_performance_plot(performance_data, plot_path,
                          metrics_col='metrics')


def parse_all_class_predictions(tasks: list[str], prediction_dir: str, reparse: bool = False):
    for task in tasks:
        
        parsing_log = os.path.join(
            prediction_dir, task.lower().replace(' ', '_'), 'parsing_log.txt')

        all_pred_files = get_all_prediction_files(prediction_dir, task)
        parsing_reports = {}
        with open(parsing_log, "a") as log_file:
            print(f"Parsing task: {task}")
            for pred_file in tqdm(all_pred_files):
                parsing_report = parse_class_predictions(
                    pred_file, task, reparse=reparse, log_file=log_file)
                if parsing_report:
                    parsing_reports[pred_file] = parsing_report

        parsing_report_path = os.path.join(
            prediction_dir, task.lower().replace(' ', '_'), 'parsing_report.json')
        if os.path.exists(parsing_report_path):
            with open(parsing_report_path, "r") as f:
                existing_reports = json.load(f)
                existing_reports.update(parsing_reports)
                parsing_reports = existing_reports
        with open(parsing_report_path, "w") as f:
            json.dump(parsing_reports, f, indent=4)


def evaluate_all_class_tasks(tasks: list[str], prediction_dir: str, reevaluate: bool = False):
    for task in tasks:
        all_pred_files = get_all_prediction_files(prediction_dir, task)
        performance_reports_path = os.path.join(
            prediction_dir, task.lower().replace(' ', '_'), 'performance_reports.json'
        )

        if os.path.exists(performance_reports_path):
            with open(performance_reports_path, "r") as f:
                performance_reports = json.load(f)
        else:
            performance_reports = {}

        for pred_file in tqdm(all_pred_files):
            print(f"Evaluating: {pred_file}")
            model = parse_file_name(pred_file, "model")
            if model in performance_reports and not reevaluate:
                print(
                    f"Skipping evaluation for {model} as it is already evaluated.")
                continue
            performance_report = evaluate_predictions(pred_file, task)
            performance_reports[model] = performance_report

        with open(performance_reports_path, "w") as f:
            json.dump(performance_reports, f, indent=4)

        # Step 3: Add BERT performance data
        performance_reports = add_bert_performance_data(
            task, performance_reports_path)

        # Step 4: Make performance plot
        plot_path = os.path.join(
            prediction_dir, task.lower().replace(' ', '_'), "performance_plot.png")
        make_performance_plot(performance_reports, plot_path,
                              metrics_col='metrics')

# TODO: not sure if here is the right place for this function
def add_tokens_nertags(bioner_path: str):
    test_path = '/home/vera/Documents/Uni/Master/Master_Thesis2.0/PsyNamic-Scale/data/ner_bio/test.csv'
    df_full = pd.read_csv(test_path)
    df_bioner = pd.read_csv(bioner_path)
    # add ner_tags if id matches
    for i, row in df_bioner.iterrows():
        id = row['id']
        if id in df_full['id'].values:
            matching_row = df_full[df_full['id'] == id]
            if not matching_row.empty:
                df_bioner.at[i, 'ner_tags'] = matching_row['ner_tags'].values[0]
    df_bioner.to_csv(bioner_path, index=False, encoding='utf-8')


def parse_all_ner_predictions(prediction_dir: str):
    log_file_path = os.path.join(prediction_dir, 'ner', 'parsing_log.txt')
    log_file = open(log_file_path, "a")

    ner_file = get_all_prediction_files(prediction_dir, 'ner')
    for file in ner_file:
        log_file.write(f"Processing file: {file}\n")
        df = pd.read_csv(file)
        # check if 'ner_tags' column exists
        if 'ner_tags' not in df.columns:
            add_tokens_nertags(file)
        if 'entities' not in df.columns:
            add_entities(file)
        parse_ner_predictions(file, reparse=False, log_file=log_file)
    log_file.close()


def get_ner_predictions_and_labels(file: str, pred_col='pred_labels', pred_entities_col='pred_entities', true_col='ner_tags', true_entities_col='entities'):
    df = pd.read_csv(file)
    predictions_bio = [ast.literal_eval(item) for item in df[pred_col]]
    predictions_entities = [ast.literal_eval(
        item) for item in df[pred_entities_col]]
    labels_bio = [ast.literal_eval(item) for item in df[true_col]]
    labels_entities = [ast.literal_eval(item)
                       for item in df[true_entities_col]]
    return predictions_bio, predictions_entities, labels_bio, labels_entities


def evaluate_all_ner(prediction_dir: str):
    ner_files = get_all_prediction_files(prediction_dir, 'ner')
    performance_reports_path = os.path.join(
        prediction_dir, 'ner', 'performance_reports.json'
    )

    if os.path.exists(performance_reports_path):
        with open(performance_reports_path, "r") as f:
            performance_reports = json.load(f)
    else:
        performance_reports = {}

    for ner_file in tqdm(ner_files):
        model = parse_file_name(ner_file, "model")
        if model in performance_reports:
            print(f"Skipping evaluation for {model} as it is already evaluated.")
            continue

        pred_bio, pred_entities, true_bio, true_entities = get_ner_predictions_and_labels(
            ner_file)

        r = bootstrap_metrics(evaluate_ner_extraction,
                              pred_entities, true_entities)
        r_bio = bootstrap_metrics(evaluate_ner_bio, pred_bio, true_bio)

        performance_reports[model] = {**r, **r_bio}

    with open(performance_reports_path, "w") as f:
        json.dump(performance_reports, f, indent=4)


def main():

    PREDICTION_DIR = 'zero_shot'

    # Parse & evaluate class predictions
    parse_all_class_predictions(TASKS, PREDICTION_DIR)
    evaluate_all_class_tasks(TASKS, PREDICTION_DIR)

    # Parse & evaluate NER predictions
    parse_all_ner_predictions(PREDICTION_DIR)
    evaluate_all_ner(PREDICTION_DIR)

    df_performance = overall_class_performance(TASKS, PREDICTION_DIR)
    # print(df_performance)
    make_performance_box_plot(df_performance, 'Zero-Shot Performance Across Tasks',
                              save_path='zero_shot/overall_performance_boxplot.png')
    make_performance_spider_plot(df_performance, 'Zero-Shot Performance Across Tasks',
                                 save_path='zero_shot/overall_performance_spiderplot.png')
    averaged_performance = df_performance.groupby(
        'model')['performance'].mean().reset_index()
    make_simple_performance_plot(
        averaged_performance, 'zero_shot/overall_performance.png')

    add_bert_performance_data('ner', os.path.join(
        PREDICTION_DIR, 'ner', 'performance_reports.json'))
    df_ner_performance = overall_ner_performance(
        PREDICTION_DIR, ['f1 overall - strict', 'f1 overall - partial', 'f1_entity_type'])
    print(df_ner_performance)
    make_performance_plot(
        df_ner_performance, save_path='zero_shot/ner/overall_ner_performance.png')


if __name__ == "__main__":
    main()
