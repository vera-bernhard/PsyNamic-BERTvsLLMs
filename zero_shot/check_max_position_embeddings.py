from transformers import AutoTokenizer
import pandas as pd
import os
import json

models = [
    'meta-llama/Llama-2-13b-chat-hf',
    'meta-llama/Meta-Llama-3-8B-Instruct',
    'meta-llama/Llama-3.1-8B-Instruct',
]

def find_file_startswith(dir: str, startswith: str) -> str:
    for filename in os.listdir(dir):
        if filename.startswith(startswith):
            return filename
    return None
# Load max position embeddings from JSON file
with open('max_position_embeddings.json', 'r') as f:
    max_position_embeddings = json.load(f)

class_description_shortened = '''Task: Classify the condition(s) of study participants from the title and abstract.
Output: Return exactly one JSON with 1/0 for each of the 14 condition options: Psychiatric condition, Depression, Anxiety, PTSD, Alcoholism, Other addictions, Anorexia, Alzheimer’s, Non-Alzheimer dementia, Substance abuse, (Chronic) Pain, Palliative Setting, Recreational Drug Use, Healthy Participants.'''

TASKS = [
    "Condition", "Data Collection", "Data Type", "Number of Participants", "Age of Participants", "Application Form",
    "Clinical Trial Phase",  "Outcomes", "Regimen", "Setting", "Study Control", "Study Purpose",
    "Substance Naivety", "Substances", "Sex of Participants", "Relevant", "Study Conclusion",
    "Study Type", "ner"
]

# Check for zero-shot prompts exceeding model context windows
# for model_name in models:
#     # print(f"Checking model {model_name}")
#     for task in TASKS:
#         # print(f'\tChecking task {task}')
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         model_short = model_name.split('/')[-1]
#         task_lower = task.lower().replace(' ', '_')
#         filename = find_file_startswith(f'zero_shot/{task_lower}', task_lower + '_' + model_short)
#         df = pd.read_csv(os.path.join(f'zero_shot/{task_lower}', filename))
#         if tokenizer.pad_token is None:
#             tokenizer.pad_token = tokenizer.eos_token
#         for prompt in df['prompt']: 
#             inputs = tokenizer(
#                         prompt, return_tensors="pt")
#             input_len = inputs['input_ids'].shape[1]
#             max_ctx = max_position_embeddings[model_name]
#             if input_len > max_ctx:
#                 print(f"Prompt length {input_len} exceeds context window ({max_ctx}) for model {model_name}. Input will be truncated, in file {filename} in row {df.index[df['prompt'] == prompt][0]}.") 
#                 print(f'ID: {df["id"][df["prompt"] == prompt].values[0]}')
    
# Check few-shot prompts
for model_name in models:
    print(f"Checking model {model_name}")
    for task in TASKS:
        for cond in ['1shot', '3shot', '5shot']:
            print(f'\tChecking task {task} with condition {cond}')
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model_short = model_name.split('/')[-1]
            task_lower = task.lower().replace(' ', '_')
            filename = find_file_startswith(f'few_shot/{task_lower}', task_lower + '_' + cond + '_selected_' + model_short)
            df = pd.read_csv(os.path.join(f'few_shot/{task_lower}', filename))
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            prompts_with_template = []
            for _, row in df.iterrows():
                prompt = row['prompt']
                # Cut off everything between ***TASK*** and ***EXAMPLES***
                # prompt_only_examples = prompt.split('***EXAMPLES***')[1]
                # prompt_only_examples += '***EXAMPLES***'
                # inputs = tokenizer(
                #         prompt_only_examples, return_tensors="pt")
                inputs = tokenizer(prompt, return_tensors="pt")
                input_len = inputs['input_ids'].shape[1]
                max_ctx = max_position_embeddings[model_name]
                if input_len > max_ctx:
                    print(f"Prompt length {input_len} exceeds context window ({max_ctx}) for model {model_name}. Input will be truncated.")
                    print(f'ID: {df["id"][df["prompt"] == prompt].values[0]}')