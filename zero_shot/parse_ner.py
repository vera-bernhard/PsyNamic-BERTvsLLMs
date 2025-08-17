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

for file in os.listdir('zero_shot/ner'):
    if not file.endswith('.csv'):
        continue
    print(f"Processing file: {file}")
    file_path = os.path.join('zero_shot/ner', file)
    parse_ner_predictions(file_path, 'entities')
    file_parts = file.split('_')
    model = file_parts[-2] + '_' + file_parts[-1].replace('.csv', '')
    print(model)
    df = pd.read_csv(file_path)
