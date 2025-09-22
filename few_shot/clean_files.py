
import os
import sys
import json
from collections import defaultdict

cur_dir = os.path.dirname(os.path.abspath(__file__))

# iterate through all files in the current directory, recursively
for root, dirs, files in os.walk(cur_dir):
    # skip the directory if its name is 'history' and 'ner'
    if 'history' in root.split(os.sep) or 'ner' in root.split(os.sep):
        continue
    for file in files:
        if file.endswith('.csv'):
            # Check if string "***FINAL INPUT TO CLASSIFY*** is in the file
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                content = f.read()
                if '***FINAL INPUT TO CLASSIFY***' in content:
                    continue
                else:
                    # prompt the user to says yes or no to moving the file into the folder parentdir/history
                    user_input = input(f"Move {file_path} to history? (yes/no): ")
                    if user_input.lower() == 'yes':
                        parentdir = os.path.dirname(root)
                        history_dir = os.path.join(parentdir, 'history')
                        # move the file to the history directory which already exists
                        os.rename(file_path, os.path.join(history_dir, file))
                        
                    else:
                        print(f"Skipping {file_path}")
                    