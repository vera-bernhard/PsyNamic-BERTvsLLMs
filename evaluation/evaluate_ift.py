import sys
import os
import json
import pandas as pd

sys.path.append(os.path.abspath('..'))
from evaluation.evaluate_zero_shot import TASKS
from evaluate_few_shot import overall_class_performance
from plots.plots import make_ift_class_performance_plot


def overall_ift_performance(prediction_dirs: str) -> pd.DataFrame:
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
                        'f1-weighted': report['metrics']['f1-weighted'][0],
                        'lower': report['metrics']['f1-weighted'][1][0],
                        'upper': report['metrics']['f1-weighted'][1][1]
                    }
                rows.append(row)
            elif model == 'Llama-3.1-8B-Instruct':
                row = {
                    'model': 'Llama-3.1-8B-Instruct',
                    'task': task,
                    'f1-weighted': report['metrics']['f1-weighted'][0],
                    'lower': report['metrics']['f1-weighted'][1][0],
                    'upper': report['metrics']['f1-weighted'][1][1]
                }
                rows.append(row)

            elif 'bert' in model.lower():
                row = {
                    'model': 'bert-baseline',
                    'task': task,
                    'f1-weighted': report['metrics']['f1-weighted'][0],
                    'lower': report['metrics']['f1-weighted'][1][0],
                    'upper': report['metrics']['f1-weighted'][1][1]
                }
                rows.append(row)
    
    df_all_tasks = overall_class_performance(TASKS, prediction_dir='few_shot')
    # remove bert baseline from df_all_tasks
    df_all_tasks = df_all_tasks[~df_all_tasks['model'].str.contains('bert', case=False)]
    df_all_tasks = df_all_tasks[~df_all_tasks['model'].str.contains('tuned', case=False)]
    # only keep highest performing model, condition combination per task
    best_models = []
    df_best = df_all_tasks.loc[df_all_tasks.groupby('task')['f1-weighted'].idxmax()]
    for _, row in df_best.iterrows():
        best_models.append({
            'model': row['model'],
            'condition': row['condition'],
            'task': row['task'],
            'f1-weighted': row['f1-weighted'],
        })
    return pd.DataFrame(rows), pd.DataFrame(best_models)
    
    

def main():
    PRED_DIRS = 'zero_shot'
    all_ift, best_models = overall_ift_performance(PRED_DIRS)
    make_ift_class_performance_plot(all_ift, best_models)

if __name__ == "__main__":
    main()
