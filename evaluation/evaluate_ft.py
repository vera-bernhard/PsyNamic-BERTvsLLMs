from evaluation.evaluate_zero_shot import TASKS
from evaluate_zero_shot import get_ner_predictions_and_labels
from parsing import add_entities
from plots.plots import make_ift_performance_plot, make_ner_error_analysis_plot
from evaluate import evaluate_ner_bio, bootstrap_metrics, ner_error_analysis, evaluate_ner_extraction
from evaluate_few_shot import overall_class_performance, overall_ner_performance
import sys
import os
import json
import pandas as pd

sys.path.append(os.path.abspath('..'))
# from evaluation.evaluate_zero_shot import TASKS


def overall_ift_performance(prediction_dirs: str, metric: str) -> pd.DataFrame:
    rows = []
    for task in TASKS:
        task_lower = task.lower().replace(' ', '_')
        with open(os.path.join(prediction_dirs, task_lower, 'performance_reports.json')) as f:
            performance_reports = json.load(f)
        for model, report in performance_reports.items():
            if 'tuned' in model:
                row = {
                    'model': 'Llama-3.1-8B-Instruct-IFT',
                    'task': task,
                    metric: report['metrics'][metric][0],
                    'lower': report['metrics'][metric][1][0],
                    'upper': report['metrics'][metric][1][1]
                }
                rows.append(row)
            elif model == 'Llama-3.1-8B-Instruct':
                row = {
                    'model': 'Llama-3.1-8B-Instruct',
                    'task': task,
                    metric: report['metrics'][metric][0],
                    'lower': report['metrics'][metric][1][0],
                    'upper': report['metrics'][metric][1][1]
                }
                rows.append(row)

            elif 'bert' in model.lower():
                row = {
                    'model': 'bert-baseline',
                    'task': task,
                    metric: report['metrics'][metric][0],
                    'lower': report['metrics'][metric][1][0],
                    'upper': report['metrics'][metric][1][1]
                }
                rows.append(row)
    # Get few-shot overall class performance
    df_all_tasks = overall_class_performance(TASKS, prediction_dir='few_shot')
    # remove bert baseline from df_all_tasks
    df_all_tasks = df_all_tasks[~df_all_tasks['model'].str.contains(
        'bert', case=False)]
    df_all_tasks = df_all_tasks[~df_all_tasks['model'].str.contains(
        'tuned', case=False)]
    # only keep highest performing model, condition combination per task
    best_models = []
    # select only rows that correspond to the requested metric name (stored in column 'metric')
    df_metric = df_all_tasks[df_all_tasks['metric'] == metric]
    df_best = df_metric.loc[df_metric.groupby('task')['mean'].idxmax()]

    for _, row in df_best.iterrows():
        best_models.append({
            'model': row['model'],
            'condition': row['condition'],
            'task': row['task'],
            metric: row['mean'],
            'lower': row['lower'],
            'upper': row['upper']
        })
    return pd.DataFrame(rows), pd.DataFrame(best_models)


def overall_ift_ner_performance(prediction_dirs: str, lst_dir: str, metric: str = 'f1 overall - strict') -> pd.DataFrame:
    rows = []
    error_rows = []
    task_lower = 'ner'
    with open(os.path.join(prediction_dirs, task_lower, 'performance_reports.json')) as f:
        performance_reports = json.load(f)
    for model, report in performance_reports.items():
        if 'tuned' in model:
            row = {
                'model': 'Llama-3.1-8B-Instruct-IFT',
                'task': 'NER',
                metric: report[metric]['mean'],
                'lower': report[metric]['lower'],
                'upper': report[metric]['upper']
            }
            rows.append(row)
            for e in report['errors']:
                error_rows.append({
                    'model': 'Llama-3.1-8B-Instruct-IFT',
                    'error_type': e,
                    'count': report['errors'][e]
                })

        elif model == 'Llama-3.1-8B-Instruct':
            row = {
                'model': 'Llama-3.1-8B-Instruct',
                'task': 'NER',
                metric: report[metric]['mean'],
                'lower': report[metric]['lower'],
                'upper': report[metric]['upper']
            }
            rows.append(row)
            for e in report['errors']:
                error_rows.append({
                    'model': 'Llama-3.1-8B-Instruct',
                    'error_type': e,
                    'count': report['errors'][e]
                })

        elif 'bert' in model.lower():
            row = {
                'model': 'bert-baseline',
                'task': 'NER',
                metric: report[metric]['mean'],
                'lower': report[metric]['lower'],
                'upper': report[metric]['upper']
            }
            rows.append(row)
            for e in report['errors']:
                error_rows.append({
                    'model': 'bert-baseline',
                    'error_type': e,
                    'count': report['errors'][e]
                })
            

    # add label supervised performance
    lst_performance_file = os.path.join(lst_dir, 'performance_report.json')
    with open(lst_performance_file) as f:
        lst_performance = json.load(f)
    row = {
        'model': 'Llama-3.1-8B-Instruct-LST',
        'task': 'NER',
        metric: lst_performance[metric]['mean'],
        'lower': lst_performance[metric]['lower'],
        'upper': lst_performance[metric]['upper']
    }
    rows.append(row)
    for e in lst_performance['errors']:
        error_rows.append({
            'model': 'Llama-3.1-8B-Instruct-LST',
            'error_type': e,
            'count': lst_performance['errors'][e]
        })


    # Get best few-shot NER performance
    overall_ner_data, error_df = overall_ner_performance(
        prediction_dir='few_shot')
    # remove bert and tuned models from overall_ner_data
    overall_ner_data = overall_ner_data[~overall_ner_data['model'].str.contains(
        'bert', case=False)]
    overall_ner_data = overall_ner_data[~overall_ner_data['model'].str.contains(
        'tuned', case=False)]
    # get best model-condition combination, based on the specified metric
    best_row = overall_ner_data.loc[
        overall_ner_data[overall_ner_data["metric"].str.lower(
        ) == metric.lower()]["mean"].idxmax()
    ]
    best_model = {
        'model': best_row['model'],
        'condition': best_row['condition'],
        'task': 'NER',
        metric: best_row['mean'],
        'lower': best_row['lower'],
        'upper': best_row['upper']
    }
    # Get corresponding error analysis
    error_df = error_df[
        (error_df['model'] == best_row['model']) &
        (error_df['condition'] == best_row['condition'])
    ]
    for _, best_row in error_df.iterrows():
        error_rows.append({
            'model': best_row['model'],
            'error_type': best_row['error_type'],
            'count': best_row['count'],
            'condition': best_row['condition']
        })
        
    return pd.DataFrame(rows), pd.DataFrame([best_model]), pd.DataFrame(error_rows)


def evaluate_lst(lst_dir: str):
    file = os.path.join(lst_dir, 'test_predictions.csv')
    pred_df = pd.read_csv(file)
    pred_labels = pred_df['pred_labels'].apply(eval).tolist()
    true_labels = pred_df['ner_tags'].apply(eval).tolist()

    # add true entities
    if 'entities' not in pred_df.columns:
        ref_file = '/home/vera/Documents/Uni/Master/Master_Thesis2.0/PsyNamic-Scale/zero_shot/ner/ner_gpt-4o-2024-08-06_22-07-22.csv'
        ref_df = pd.read_csv(ref_file)
        pred_df = pred_df.merge(ref_df[['id', 'entities']], on='id')
        # write back to file
        pred_df.to_csv(file, index=False)

    if 'pred_entities' not in pred_df.columns:
        add_entities(file, bio_col='pred_labels', output_col='pred_entities')
        pred_df = pd.read_csv(file)  # reload to get updated entities column

    # rename

    pred_bio, pred_entities, true_bio, true_entities = get_ner_predictions_and_labels(
        file)
    r_bio = bootstrap_metrics(evaluate_ner_bio, pred_bio, true_bio)
    r = bootstrap_metrics(evaluate_ner_extraction,
                          pred_entities, true_entities)

    e = ner_error_analysis(pred_bio, true_bio)

    # write into json file
    results = {}
    results.update(r)
    results.update(r_bio)
    results['errors'] = e
    with open(os.path.join(lst_dir, 'performance_report.json'), 'w') as f:
        json.dump(results, f, indent=4)


def main():
    PRED_DIRS = 'zero_shot'
    OUTPUT_DIR = 'finetuning'
    LST_DIR = 'finetuning/lst_llama3_8b_2.0/'
    # evaluate_lst(LST_DIR)
    all_ift, best_models = overall_ift_performance(
        PRED_DIRS, metric='f1-weighted')
    make_ift_performance_plot(all_ift, best_models, title='Best ICL vs. IFT vs. BERT for Classification',
                                    save_path=os.path.join(OUTPUT_DIR, 'ift_class_performance.png'))
    all_ift_ner, best_model_ner, error_data = overall_ift_ner_performance(
        PRED_DIRS, LST_DIR,  metric='f1 overall - strict')
    make_ift_performance_plot(all_ift_ner, best_model_ner, title='Best ICL vs. IFT vs. LST vs. BERT for NER',
                                    save_path=os.path.join(OUTPUT_DIR, 'ift_ner_performance.png'), metric='f1 overall - strict')
    make_ner_error_analysis_plot(error_data, 'BEST ICL vs. IFT vs. LST vs. BERT for NER - Error Analysis',
                                save_path=os.path.join(OUTPUT_DIR, 'ift_ner_error_analysis.png'))
    
    all_ift_ner, best_model_ner, error_data = overall_ift_ner_performance(
        PRED_DIRS, LST_DIR,  metric='f1_overall')
    make_ift_performance_plot(all_ift_ner, best_model_ner, title='Best ICL vs. IFT vs. LST vs. BERT for NER',
                                    save_path=os.path.join(OUTPUT_DIR, 'ift_ner_performance_f1_overall.png'), metric='f1_overall')


if __name__ == "__main__":
    main()
