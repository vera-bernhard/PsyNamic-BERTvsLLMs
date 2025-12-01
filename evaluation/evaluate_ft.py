from evaluation.evaluate_zero_shot import TASKS
from evaluate_zero_shot import get_ner_predictions_and_labels
from parsing import add_entities
from plots.plots import make_ift_performance_plot, make_ner_error_analysis_plot, size_plot, medical_vs_general_plot, ift_box_plot
from evaluate import evaluate_ner_bio, bootstrap_metrics, ner_error_analysis, evaluate_ner_extraction
from evaluate_few_shot import overall_class_performance, overall_ner_performance
import sys
import os
import json
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath('..'))
# from evaluation.evaluate_zero_shot import TASKS


def overall_ift_performance(zero_shot_dir: str, few_shot_dir: str, metric: str) -> pd.DataFrame:
    rows = []
    for task in TASKS:
        task_lower = task.lower().replace(' ', '_')
        with open(os.path.join(zero_shot_dir, task_lower, 'performance_reports.json')) as f:
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
    df_all_tasks = overall_class_performance(
        TASKS, prediction_dir=few_shot_dir)
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

    # get best performance over 0, 1, 3, 5 shots for gpt-4o-2024-08-06
    df_gpt = df_metric[df_metric['model'] == 'gpt-4o-2024-08-06']
    for taksk in TASKS:
        df_task = df_gpt[df_gpt['task'] == taksk]
        if not df_task.empty:
            best_row = df_task.loc[df_task['mean'].idxmax()]
            rows.append({
                'model': best_row['model'],
                'condition': best_row['condition'],
                'task': best_row['task'],
                metric: best_row['mean'],
                'lower': best_row['lower'],
                'upper': best_row['upper']
            })

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


def overall_ift_ner_performance(zero_shot_dir: str, lst_dir: str, few_shot_dir: str, metric: str = 'f1 overall - strict') -> pd.DataFrame:
    rows = []
    error_rows = []
    task_lower = 'ner'
    with open(os.path.join(zero_shot_dir, task_lower, 'performance_reports.json')) as f:
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
        prediction_dir=few_shot_dir)
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
    nr_empty_with_pred = 0
    nr_empty_without_pred = 0
    for p, t in zip(pred_entities, true_entities):
        if len(t) == 0 and len(p) > 0:
            nr_empty_with_pred += 1
        if len(t) == 0 and len(p) == 0:
            nr_empty_without_pred += 1
    e['nr_empty_with_pred'] = nr_empty_with_pred
    e['nr_empty_without_pred'] = nr_empty_without_pred
    # write into json file
    results = {}
    results.update(r)
    results.update(r_bio)
    results['errors'] = e
    with open(os.path.join(lst_dir, 'performance_report.json'), 'w') as f:
        json.dump(results, f, indent=4)


def overall_performance(zero_shot_dir: str, lst_dir: str, few_shot_dir: str):
    all_ift, _ = overall_ift_performance(
        zero_shot_dir, few_shot_dir, metric='f1-weighted')
    # all_ift_ner, _, _ = overall_ift_ner_performance(
    #     zero_shot_dir, lst_dir, few_shot_dir, metric='f1 overall - strict')
    all_ift = all_ift.rename(columns={'f1-weighted': 'metric'})
    # all_ift_ner = all_ift_ner.rename(columns={'f1 overall - strict': 'metric'})
    # merged = pd.concat([all_ift, all_ift_ner], ignore_index=True)
    merged = all_ift.copy()
    # remove: rows with Llama-3.1-8B-Instruct
    merged = merged[merged['model'] != 'Llama-3.1-8B-Instruct']

    # add condition column
    merged['condition'] = ''
    merged.loc[merged['model'] ==
               'Llama-3.1-8B-Instruct-IFT', 'condition'] = 'fine-tuned'
    # rename 'llama-3.1-8b-instruct-ift' to 'llama-3.1-8b-instruct'
    merged.loc[merged['model'] == 'Llama-3.1-8B-Instruct-IFT', 'model'] = 'Llama-3.1-8B-Instruct'
    # all BERT baseline condition == 'BERT'
    merged.loc[merged['model'] ==
               'bert-baseline', 'condition'] = 'fine-tuned'

    all_ner_few, all_error_few = overall_ner_performance(few_shot_dir)
    all_ner_few['task'] = 'NER'
    all_class_few = overall_class_performance(TASKS, few_shot_dir)

    # remove bert and tuned models from few-shot data
    all_ner_few = all_ner_few[~all_ner_few['model'].str.contains(
        'bert', case=False)]
    all_ner_few = all_ner_few[~all_ner_few['model'].str.contains(
        'tuned', case=False)]
    all_class_few = all_class_few[~all_class_few['model'].str.contains(
        'bert', case=False)]
    all_class_few = all_class_few[~all_class_few['model'].str.contains(
        'tuned', case=False)]

    for task in TASKS:
        # get zero_shot performance for this task, for all models, condition == 'zero-shot' and task == task
        rows = all_class_few[(all_class_few['task'] == task) & (all_class_few['metric'] == 'f1-weighted')]

        # rows where condition == 'zero-shot'
        zero_shot = rows[rows['condition'] == 'zero_shot']
        few_shot = rows[rows['condition'] != 'zero_shot']
        # add zero-shot rows
        for _, row in zero_shot.iterrows():
            merged = pd.concat([merged, pd.DataFrame([{
                'model': row['model'],
                'task': task,
                'metric': row['mean'],
                'lower': row['lower'],
                'upper': row['upper'],
                'condition': 'zero-shot'
            }])], ignore_index=True)
        
        # add highest score per model
        rows = few_shot.loc[few_shot.groupby('model')['mean'].idxmax()]
        for _, row in rows.iterrows():
            merged = pd.concat([merged, pd.DataFrame([{
                'model': row['model'],
                'task': task,
                'metric': row['mean'],
                'lower': row['lower'],
                'upper': row['upper'],
                'condition': 'few-shot'
            }])], ignore_index=True)

    # add best-shot NER
    rows = all_ner_few[all_ner_few['metric'] == 'f1_overall']

    zero_shot = rows[rows['condition'] == 'zero_shot']
    few_shot = rows[rows['condition'] != 'zero_shot']
    rows = rows.loc[rows.groupby('model')['mean'].idxmax()]
    for _, row in zero_shot.iterrows():
        merged = pd.concat([merged, pd.DataFrame([{
            'model': row['model'],
            'task': 'NER',
            'metric': row['mean'],
            'lower': row['lower'],
            'upper': row['upper'],
            'condition': 'zero-shot'
        }])], ignore_index=True)
    # get best few-shot per model
    rows = few_shot.loc[few_shot.groupby('model')['mean'].idxmax()]
    for _, row in rows.iterrows():
        merged = pd.concat([merged, pd.DataFrame([{
            'model': row['model'],
            'task': 'NER',
            'metric': row['mean'],
            'lower': row['lower'],
            'upper': row['upper'],
            'condition': 'few-shot'
        }])], ignore_index=True)

    # add model size column
    model_size_map = {
        'Llama-3.1-8B-Instruct': 8,
        'Meta-Llama-3-8B-Instruct': 8,
        'Med-LLaMA3-8B': 8,
        'MeLLaMA-70B-chat': 70,
        'MeLLaMA-13B-chat': 13,
        'Llama-2-70b-chat-hf': 70,
        'Llama-2-13b-chat-hf': 13,
        'gemma-3-27b-it': 27,
        'medgemma-27b-text-it': 27,
        'gpt-4o-2024-08-06': 200,
        'gpt-4o-mini': 30,
        'bert-baseline': 0.11,
        'Llama-3.1-8B-Instruct-IFT': 8.3,
    }
    merged['model_size'] = merged['model'].map(model_size_map)
    size_plot(merged, title='Model Size vs. Performance',
              save_path='finetuning/model_size_performance.png')
    medical_vs_general_plot(merged, title='Medical vs. General Models Performance',
                            save_path='finetuning/medical_vs_general_performance.png')
    return merged


def ift_stat_class(ift_df: pd.DataFrame, best_models: pd.DataFrame):
    # average improvement from llama-3.1-8b-instruct to llama-3.1-8b-instruct-ift
    if_df = ift_df[ift_df['model'].isin(
        ['Llama-3.1-8B-Instruct', 'Llama-3.1-8B-Instruct-IFT'])]
    ift_diff = []
    for task in if_df['task'].unique():
        task_df = if_df[if_df['task'] == task]
        instruct_row = task_df[task_df['model']
                              == 'Llama-3.1-8B-Instruct'].iloc[0]
        ift_row = task_df[task_df['model']
                          == 'Llama-3.1-8B-Instruct-IFT'].iloc[0]
        improvement = ift_row['f1-weighted'] - instruct_row['f1-weighted']
        ift_diff.append(improvement)
        print(
            f'Task: {task}, Improvement from ICL to IFT: {improvement:.4f}')
        
    avg_improvement = sum(ift_diff) / len(ift_diff)
    print(f'Average Improvement Llama-3.1 to IFT across tasks: {avg_improvement:.4f}')
    # median improvement
    median_improvement = sorted(ift_diff)[len(ift_diff) // 2]
    print(f'Median Improvement Llama-3.1 to IFT across tasks: {median_improvement:.4f}')
    print(f'Median performance of IFT: {sorted([row["f1-weighted"] for _, row in if_df[if_df["model"] == "Llama-3.1-8B-Instruct-IFT"].iterrows()])[len(ift_diff) // 2]:.4f}')
    print(f'Median performance of zero-shot: {sorted([row["f1-weighted"] for _, row in if_df[if_df["model"] == "Llama-3.1-8B-Instruct"].iterrows()])[len(ift_diff) // 2]:.4f}')
    print(f'Median peformance of ber')

    print(f'Median performance of best icl: {sorted([row["f1-weighted"] for _, row in best_models.iterrows()])[len(ift_diff) // 2]:.4f}')
    all_gpto_perf = ift_df[ift_df['model'] == 'gpt-4o-2024-08-06']
    print('Median performance of gpt-4o-2024-08-06:', all_gpto_perf['f1-weighted'].median())

    print('Average improvement from best icl to ift:')
    best_to_ift_diff = []
    for task in if_df['task'].unique():
        ift_row = if_df[(if_df['task'] == task) & (
            if_df['model'] == 'Llama-3.1-8B-Instruct-IFT')].iloc[0]
        best_row = best_models[best_models['task'] == task].iloc[0]
        improvement = ift_row['f1-weighted'] - best_row['f1-weighted']
        best_to_ift_diff.append(improvement)
        print(
            f'Task: {task}, Improvement from Best ICL to IFT: {improvement:.4f}')
    avg_best_to_ift_improvement = sum(best_to_ift_diff) / len(best_to_ift_diff)
    print(f'Average Improvement Best ICL to IFT across tasks: {avg_best_to_ift_improvement:.4f}')
    print(f'Median Improvement Best ICL to IFT across tasks: {sorted(best_to_ift_diff)[len(best_to_ift_diff) // 2]:.4f}')



def main():
    ZERO_SHOT_DIR = 'zero_shot'
    FEW_SHOT_DIR = 'few_shot'
    OUTPUT_DIR = 'finetuning'
    LST_DIR = 'finetuning/lst_llama3_8b_2.0/'
    evaluate_lst(LST_DIR)
    all_ift, best_models = overall_ift_performance(
        ZERO_SHOT_DIR, FEW_SHOT_DIR, metric='f1-weighted')
    ift_stat_class(all_ift, best_models)

    ift_box_plot(all_ift, title='Best ICL vs. IFT vs. BERT over All Classification Tasks',
                 save_path=os.path.join(OUTPUT_DIR, 'ift_boxplot_performance.png'))

    make_ift_performance_plot(all_ift, best_models, title='Best ICL vs. IFT vs. BERT for Classification',
                                    save_path=os.path.join(OUTPUT_DIR, 'ift_class_performance.png'))
    all_ift_ner, best_model_ner, error_data = overall_ift_ner_performance(
       ZERO_SHOT_DIR, LST_DIR,  FEW_SHOT_DIR, metric='f1 overall - strict')
    make_ift_performance_plot(all_ift_ner, best_model_ner, title='Best ICL vs. IFT vs. LST vs. BERT for NER',
                                    save_path=os.path.join(OUTPUT_DIR, 'ift_ner_performance.png'), metric='f1 overall - strict')
    make_ner_error_analysis_plot(error_data, 'BEST ICL vs. IFT vs. LST vs. BERT for NER - Error Analysis',
                               save_path=os.path.join(OUTPUT_DIR, 'ift_ner_error_analysis.png'))

    all_ift_ner, best_model_ner, error_data = overall_ift_ner_performance(
        ZERO_SHOT_DIR, LST_DIR,  FEW_SHOT_DIR, metric='f1_overall')
    make_ift_performance_plot(all_ift_ner, best_model_ner, title='Best ICL vs. IFT vs. LST vs. BERT for NER',
                                    save_path=os.path.join(OUTPUT_DIR, 'ift_ner_performance_f1_overall.png'), metric='f1_overall')

    overall_performance(ZERO_SHOT_DIR, LST_DIR, FEW_SHOT_DIR)
  

if __name__ == "__main__":
    main()
