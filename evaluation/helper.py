from typing import Literal
import os
import re
import unicodedata

def parse_file_name(pred_path: str, info: Literal["model", "condition", "date", "task"]) -> str:
    """Parses the prediction file name to extract model, condition, date or task.
    Example file names: 
    - setting_MeLLaMA-13B-chat_06-09-06.csv
    - clinical_trial_phase_1shot_selected_gpt-4o-2024-08-06_09-09-09.csv
    - sex_of_participants_gpt-4o-mini_09-09-09.csv
    """
    filename = os.path.basename(pred_path).replace('.csv', '')
    parts = filename.rsplit('_')

    if 'shot' in filename:
        date = parts[-1]
        model = parts[-2]
        # parts -3 and -4
        condition = parts[-3] + '_' + parts[-4]
        if info == "date":
            return date
        elif info == "model":
            return model
        elif info == "condition":
            return condition
        elif info == "task":
            return '_'.join(parts[:-3])
        else:
            raise ValueError(f"Unknown info type: {info}")
    else:
        date = parts[-1]
        model = parts[-2]
        if info == "date":
            return date
        elif info == "model":
            return model
        elif info == "task":
            return '_'.join(parts[:-2])
        else:
            raise ValueError(f"Unknown info type: {info}")

    raise ValueError(f"Filename does not match expected patterns: {filename}")



def normalize_spaces(text: str):
    """
    Clean text by:
    - Normalizing unicode characters
    - Replacing non-standard spaces (thin spaces, NBSP, zero-width, etc.) with normal space
    - Collapsing multiple spaces into one
    - Stripping leading/trailing spaces
    """
    # Normalize unicode (full-width chars → standard)
    text = unicodedata.normalize("NFKC", text)
    
    # Replace non-standard spaces with normal space
    text = re.sub(r"[\u00A0\u2000-\u200B\u202F\u205F\u3000]", " ", text)
    
    # Collapse multiple consecutive whitespace into single space
    text = re.sub(r"\s+", " ", text)
    
    # Remove leading/trailing spaces
    text = text.strip()
    
    return text

def get_substring_matches(tokens: list[str], text: str) -> list[str]:
    """
    Finds all substrings in text that start with the sequence of tokens
    (with optional whitespace between tokens) and end with the last token.
    Returns a list of matched substrings.
    """
    tokens = [normalize_spaces(token) for token in tokens]
    pattern = r'\s*'.join(map(re.escape, tokens))
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    return matches

def get_closest_match(search_string: str, options: list[str]) -> str:
    """
    Finds closest match = only varying in the use of spaces (more or less spaces)
    e.g. options ['5, 10, and 20 μg', '5 days. Plasma levels of LSD and subjective effects were assessed up to 6 hours after administration. Pharmacokinetic parameters were determined using', '5% confidence interval) maximal LSD concentrations were 151 pg', '500 pg', '5, 10, and 20 μg', '5-6.2). The 5 μg', '5 hours, and ended at 5.1 hours. The 20 μg', '5-20 μg']
    search_string = '5, 10, and 20 μg'
    --> should retun '5, 10, and 20 μg'
    """

    if search_string in options:
        return search_string
    
    # Normalize by stripping all spaces
    target = normalize_spaces(re.sub(r"\s+", "", search_string))
    for option in options:
        if normalize_spaces(re.sub(r"\s+", "", option)) == target:
            print(f"Matched '{search_string}' to '{option}' by ignoring spaces.")
            return option
    
    return None

 
