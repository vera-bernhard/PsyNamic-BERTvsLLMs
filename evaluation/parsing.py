from typing import TextIO
import re
import json
import os
import pandas as pd
import unicodedata
from typing import Literal
import spacy
import ast
from collections import Counter

nlp = spacy.load("en_core_web_sm")


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


def check_label_synonyms(label: str) -> str:
    """Hand-crafted obvious synonyms that systematically occur in llm predictions."""
    synonyms = {
        "Alzheimer's disease": "Alzheimer’s disease",
        "Alzheimer\\u2019s disease": "Alzheimer’s disease",
        ">=1000": "≥1000",
        "Substance-Naive Participants": "Substance-Naïve Participants",
        "Substance-non-naive participants": "Substance-non-naïve participants"
    }
    try:
        return synonyms[label]
    except KeyError:
        return None


def clean_token(token: str) -> str:
    """Clean the token by removing newlines and trailing punctuation."""
    token = token.strip().replace('\n', '')
    if token.endswith('.^'):
        token = token[:-2].strip()
    return token


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


def basic_tokenizer(text: str) -> list[str]:
    text = text.strip()
    doc = nlp(text)
    tokenized = [token.text for token in doc]
    return tokenized

# TODO: Check if this is somewhat the same as get_closest_match scenario


def find_phrase_indices(tokens: list[str], phrase: str) -> list[int]:
    tokens_lower = [clean_token(t.lower()) for t in tokens]
    phrase_tokens = [t.lower() for t in basic_tokenizer(phrase)]
    phrase_len = len(phrase_tokens)

    for i in range(len(tokens_lower) - phrase_len + 1):
        if tokens_lower[i:i + phrase_len] == phrase_tokens:
            return list(range(i, i + phrase_len))

    return []


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
            print(
                f"Matched '{search_string}' to '{option}' by ignoring spaces.")
            return option

    return None


def parse_ner_prediction(pred: str, tokens: list[str], text: str, log_file: TextIO = None) -> list[str]:
    """ Convert llm prediction text with <span> tags into a BIO sequence.

    The prediction contains entities like:
        <span class="application-area">depression</span>
        <span class="dosage">50 mg</span>

    Steps:
    1. Clean the prediction text depending on the model (Llama-2 / MeLLaMA).
    2. Extract spans via regex: (entity text, entity type).
    3. Align each span with tokens:
       - First token → B-<EntityLabel>, rest → I-<EntityLabel>.
       - Restore from backup if partial match fails.
       - Fallback: search span text directly in tokens.
    4. Warn if number of B- labels doesn’t match number of spans.
    """
    # if text is nan
    if pd.isna(text) or text == '':
        return ['O'] * len(tokens), []

    # TODO: Redunancy with function to extract entities
    ner_labels = {
        'application-area': 'Application area',
        'dosage': 'Dosage',
    }
    text = normalize_spaces(text)
    tokens = [normalize_spaces(t) for t in tokens]

    bio_tokens = ['O'] * len(tokens)
    token_pointer = 0

    # TODO: Don't do this twice
    spans = parse_ner_prediction_entities(pred)

    # Remove all spans with entity type not in ner_labels
    original_span_count = len(spans)
    removed_spans = [(e, t) for e, t in spans if t not in ner_labels]
    spans = [(e, t) for e, t in spans if t in ner_labels]
    removed_count = original_span_count - len(spans)
    if removed_count > 0:
        if log_file is not None:
            log_file.write(
                f"Removed {removed_count} spans with unknown entity types:\n")
            for e, t in removed_spans:
                log_file.write(f"  Removed entity: '{e}' (type: {t})\n")
        else:
            print(
                f"Removed {removed_count} spans with unknown entity types:")
            for e, t in removed_spans:
                print(f"  Removed entity: '{e}' (type: {t})")

    # Remove any spans that there string don't appear in the text at all, case insensitive
    original_span_count = len(spans)
    removed_spans = [(e, t) for e, t in spans if e.lower() not in text.lower()]
    spans = [(e, t) for e, t in spans if e.lower() in text.lower()]
    removed_count = original_span_count - len(spans)
    if removed_count > 0:
        if log_file is not None:
            log_file.write(
                f"Removed {removed_count} spans that don't appear in the text:\n")
            for e, t in removed_spans:
                log_file.write(f"  Removed entity: '{e}' (type: {t})\n")
        else:
            print(
                f"Removed {removed_count} spans that don't appear in the text:")
            for e, t in removed_spans:
                print(f"  Removed entity: '{e}' (type: {t})")

    # Iterate through the spans and in parallel through the tokens to find matches
    # - assuming that the spans are in the same order as the tokens
    # - assuming that the spans are not overlapping (which is the case for the PsyNamic Scale)
    for e, t in spans:
        e_in_bio = False  # Keep track if entity is matched
        temp_e = e.lower()  # Keep track what of the entity is still to be matched
        token_start = False  # Keep track of whether B- or I- token need to be set
        # Backup bio_tokens to reset if the beginning of the entity seemed a match but later turned out not to be
        backup_bio_tokens = bio_tokens.copy()

        # Iterate through the tokens to find matches
        for i in range(token_pointer, len(tokens)):
            if tokens[i] == '\n':
                continue
            token = tokens[i].strip().lower().removesuffix('^')
            if not token_start:
                if temp_e.startswith(token):
                    bio_tokens[i] = f'B-{ner_labels[t]}'
                    token_start = True
                    temp_e = temp_e.removeprefix(token).strip()
                    e_in_bio = True
            else:
                if temp_e.startswith(token):
                    bio_tokens[i] = f'I-{ner_labels[t]}'
                    temp_e = temp_e.removeprefix(token).strip()

                elif temp_e.startswith(clean_token(token)):
                    bio_tokens[i] = f'I-{ner_labels[t]}'
                    temp_e = temp_e.removeprefix(clean_token(token)).strip()

                else:
                    # Case 1: all of the entity is matched or only non-letter characters and only then save in bio_tokens
                    # or (len(temp_e) == 1 and not temp_e.isalnum()):
                    if temp_e == '':
                        token_start = False
                        token_pointer = i
                        break  # Move to next entity

                    # Case 2: Only part of the entity is matched --> entity must be later in the text or be faulty
                    else:
                        e_in_bio = False
                        bio_tokens = backup_bio_tokens.copy()
                        token_start = False
                        temp_e = e.lower()

        # In cases, of very messy prediction, where
        # - the order of spans in predictions is not the same as they appear in tokens
        # - the prediction text is not the same as the tokens but spans are still in text
        if not e_in_bio:
            ids = find_phrase_indices(tokens, e)
            # set first token to B- and the rest to I-
            # Check that bio_token at ids are still 'O'
            if ids:
                # Only set Bio if nothing previously set
                if all(bio_tokens[i] == 'O' for i in ids):
                    bio_tokens[ids[0]] = f'B-{ner_labels[t]}'
                    for j in ids[1:]:
                        bio_tokens[j] = f'I-{ner_labels[t]}'
            else:
                pass

    # Check if there is as many entities in bio_token as in spans
    num_entities = sum(1 for label in bio_tokens if label.startswith('B-'))
    if num_entities > len(spans):
        warning_msg = (
            f"Warning: Number of entities in bio_token ({num_entities}) is greater than number of spans ({len(spans)}).\n"
        )
        if log_file is not None:
            log_file.write(warning_msg)
        else:
            print(warning_msg)

    if num_entities != len(spans):
        warning_msg = (
            f"Warning: Number of entities in bio_token ({num_entities}) does not match number of spans ({len(spans)}).\n"
        )
        if log_file is not None:
            log_file.write(warning_msg)
        else:
            print(warning_msg)
    if log_file is not None:
        log_file.flush()

    return bio_tokens, spans


def parse_ner_prediction_entities(pred: str) -> list[tuple[str, str]]:
    spans = re.findall(r'<span class="(.*?)">(.*?)</span>', pred)
    spans = [(e, t) for t, e in spans]
    spans = [(e.strip(), t.strip('"')) for e, t in spans]
    return spans


def is_one_hot(string: str, length: int) -> bool:
    """Check if a string is a one-hot encoded list of given length.
    It recognizes cases like "0 0 1 0", "0,0,1,0", "[0, 0, 1, 0]", etc.
    """
    string = string.strip().lstrip("[").rstrip("]")

    if "," in string:
        parts = [x.strip() for x in string.split(",")]
    else:
        parts = [x.strip() for x in string.split()]

    if len(parts) != length:
        return False

    return all(part in {"0", "1"} for part in parts)


def is_first_line_one_hot(string: str, length: int) -> bool:
    """Check if the first line of a string is a one-hot encoded list of given length."""
    first_line = string.split('\n')[0].strip()
    return is_one_hot(first_line, length)


def get_entity_string_from_bio(tokens: list[str], sample_text: str):
    """ Convert a list of tokens that correspond to one entity, into a string which occurs in the sample text.
        e.g. ['5', ',', '10', ',', 'and', '20', 'μg'] --> "5, 10, and 20μg"
        This is necesary because after tokenization the information about spaces is lost.    
    """
    if not tokens:
        return "", None

    merged = " ".join(tokens)

    replacements = [
        (" ,", ","), (" .", "."), (" :", ":"), (" ;", ";"),
        (" )", ")"), ("( ", "("), (" '", "'"), ("`` ", "``"),
        (" ''", "''"), (" - ", "-"), (" / ", "/"), (" %", "%"),
        ("$ ", "$"), (" @ ", "@"), (" # ", "#"), (" & ", "&"),
        (" * ", "*"), (" ...", "...")
    ]
    for old, new in replacements:
        merged = merged.replace(old, new)

    if sample_text:
        sample_text = normalize_spaces(sample_text)
        if merged not in sample_text:
            options = get_substring_matches(tokens, sample_text)
            merged = get_closest_match(merged, options)
            if not merged:
                pass
    return merged


def get_substring_matches(tokens: list[str], text: str) -> list[str]:
    """
    Finds all substrings in `text` that match the sequence of `tokens`, allowing for variable spacing.
    """
    tokens = [normalize_spaces(token) for token in tokens]
    pattern = r'\s*'.join(map(re.escape, tokens))
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    return matches


def parse_class_prediction(pred_text: str, label2int: dict, model: str) -> tuple[str, bool]:
    """Parse the prediction text from various generative llms into a one-hot encoded list of labels.

    """
    # if pred text is nan
    if pd.isna(pred_text) or pred_text == '':
        raise ValueError("Prediction text is empty or NaN.")

    # TODO: check if this is even necessary
    if model.startswith('Llama-2'):
        # Split at [/INST]
        parts = pred_text.split('[/INST]')
        pred_text = parts[-1].strip()

    elif model.startswith('MeLLaMA'):
        parts = pred_text.split('OUTPUT:')
        # if len(parts) != 2:
        #     breakpoint()
        #     raise ValueError(
        #         f'Prediction text does not contain "OUTPUT:": {pred_text}')
        pred_text = parts[-1].strip()

    elif model.startswith('gpt'):
        pred_text = pred_text.replace('\\n', '\n')

    elif model.startswith('Med-LLaMA3'):
        # remove INPUT and following text
        parts = pred_text.split('INPUT:')
        pred_text = parts[0].strip()

    faulty_but_parsable = False
    if '{' in pred_text:
        # Case 1: There is a prediction in dictionary format
        start = pred_text.index('{')
        try:
            end = pred_text.index('}')
        # There is no closing bracket
        except ValueError:
            # Add closing bracket at the end
            pred_text += '}'
            end = pred_text.index('}')
            faulty_but_parsable = True
        prediction_dict = pred_text[start:end+1]
        # Clean up the prediction dictionary so that is valid JSON
        prediction_dict = prediction_dict.replace('""', '"')
        prediction_dict = re.sub(r':\s*"\s*(?=[,}])', ': ""', prediction_dict)
        prediction_dict = json.loads(prediction_dict)

        # Check if there is empty predictions -> "" instead of 0 or 1
        if "" in prediction_dict.values():
            faulty_but_parsable = True
            # Check if there is at leas one 1
            if not any(value == 1 or value == '1' for value in prediction_dict.values()):
                print(f"Not parsable: {pred_text}")
                faulty_but_parsable = False
                raise ValueError(f'Could not parse prediction: {pred_text}')
            else:
                # replace empty predictions with 0
                for key in prediction_dict.keys():
                    if prediction_dict[key] == "":
                        prediction_dict[key] = 0

        # Create one-hot encoding, according to label2int
        onehot_list = [0] * len(label2int)
        for label, value in prediction_dict.items():
            # Sanity check: label exists in label2int
            if label not in label2int:
                label_syn = check_label_synonyms(label)
                if label_syn is not None:
                    label = label_syn
                else:
                    raise ValueError(
                        f'Label {label} not found in label2int mapping.')
            pos = label2int[label]
            onehot_list[pos] = int(value)
        return str(onehot_list), faulty_but_parsable

    elif pred_text in label2int.keys():
        faulty_but_parsable = True
        # Case 2: There is a prediction in string format, e.g. 'Randomized-controlled trial (RCT)'
        onehot_list = [0] * len(label2int)
        pos = label2int[pred_text]
        onehot_list[pos] = 1
        return str(onehot_list), faulty_but_parsable

    elif is_one_hot(pred_text, len(label2int)):
        faulty_but_parsable = True
        # Case 4: There is a prediction in one-hot format with valid length, e.g. '0, 1, 0, 0, 0, 0, 0, 0, 0'
        if ',' in pred_text:
            onehot_list = [int(x.strip()) for x in pred_text.split(",")]
        else:
            onehot_list = [int(x.strip()) for x in pred_text.split()]
        return str(onehot_list), faulty_but_parsable

    elif is_first_line_one_hot(pred_text, len(label2int)):
        faulty_but_parsable = True
        # Case 5: There is a prediction in one-hot format with valid length in the first line, e.g. '0, 1, 0, 0, 0, 0, 0, 0, 0\nSome explanation'
        first_line = pred_text.split('\n')[0].strip()
        if ',' in first_line:
            onehot_list = [int(x.strip()) for x in first_line.split(",")]
        else:
            onehot_list = [int(x.strip()) for x in first_line.split()]
        return str(onehot_list), faulty_but_parsable

    elif ':' in pred_text:
        if pred_text.count(':') > 1:
            preds = pred_text.split('\n')
            # remove empty lines
            preds = [p for p in preds if p.strip() != '']
            onehot_list = [0] * len(label2int)
            for p in preds:
                label = p.split(':')[0].strip()
                value = p.split(':')[1].strip()
                if label not in label2int:
                    label_syn = check_label_synonyms(label)
                    if label_syn is not None:
                        label = label_syn
                    else:
                        raise ValueError(
                            f'Label {label} not found in label2int mapping.')
                pos = label2int[label]
                onehot_list[pos] = int(value)
            faulty_but_parsable = True
            return str(onehot_list), faulty_but_parsable

        faulty_but_parsable = True
        # Case 3: There is a prediction in string format with a score, e.g. 'Randomized-controlled trial (RCT): 1
        onehot_list = [0] * len(label2int)
        label = pred_text.split(':')[0].strip()
        pos = label2int[label]
        onehot_list[pos] = 1
        return str(onehot_list), faulty_but_parsable

    else:
        raise ValueError(f'Could not parse prediction: {pred_text}')
    

def convert_berttoken_to_bio(file: str, other_pred_file: str) -> str:
    """
    Convert a CSV file with id, token, prediction columns into 
    a list of BIO labels and entities, merging subwords (##).

    Example input:
        id,token,word_id,prediction,probability,label
        2431,default,0,O,"[0.9994654059410095, 2.1159517928026617e-05, 5.5784756114007905e-05, 4.1225506720365956e-05, 0.000416381168179214]",O
        2431,mode,1,O,"[0.9995869994163513, 2.3245402189786546e-05, 0.00030319008510559797, 2.5240009563276544e-05, 6.131161353550851e-05]",O
        2431,connectivity,2,O,"[0.9998838901519775, 8.697813427716028e-06, 4.1984756535384804e-05, 1.6123200111906044e-05, 4.937177072861232e-05]",O
        2431,in,3,O,"[0.9998694658279419, 1.8335587810724974e-05, 5.0957842177012935e-05, 1.7795697203837335e-05, 4.3426607589935884e-05]",O
        2431,major,4,B-Application area,"[0.027369728311896324, 0.0006279110675677657, 0.00217921263538301, 0.000719171017408371, 0.969103991985321]",B-Application area

    Example Output:
    id,text,prompt,prediction_text,model,tokens,pred_labels,ner_tags,pred_entities,entities
    
    """
    df = pd.read_csv(file)
    df_pred = pd.read_csv(other_pred_file)

    model = os.path.basename(file).split('_')[0]
   
    new_records = []

    ids = df['id'].unique()
    for id in ids:
        # get all rows keeping the order of the rows
        df_id = df[df['id'] == id]
        df_id = df_id.reset_index(drop=True)
       
        # get tokens from df_pred
        match = df_pred[df_pred['id'] == id]
        if len(match) != 1:
            raise ValueError(
                f"Could not find unique row in {other_pred_file} with id {id}")
        match_row = match.iloc[0]
        
        # Check that number of unique word_ids matches number of tokens
        max_word_id = df_id['word_id'].max()+1
        tokens = eval(match_row['tokens']) 
        if max_word_id != len(tokens):
            raise ValueError(
                f"Number of unique word_ids {max_word_id} does not match number of tokens {len(tokens)} in {other_pred_file} with id {id}")
        new_bio_labels = len(tokens) * ['O']

        current_word_id = 0
        word_id_labels = []
        for i, row in df_id.iterrows():
            label = row['prediction']
            
            # New word has started, save previous word_id labels
            if row['word_id'] != current_word_id:
                # Save previous word_id labels
                if len(set(word_id_labels)) == 1:
                    new_bio_labels[current_word_id] = word_id_labels[0]
                    previous_label = word_id_labels[0]
                    word_id_labels = []
                    
                elif len(set(word_id_labels)) > 1:
                    # Check if there is 1 B- and rest I- labels and same type
                    label_types = set(l[2:] for l in word_id_labels if l != 'O')
                    if len(label_types) == 1 and any(l.startswith('B-') for l in word_id_labels):
                        new_bio_labels[current_word_id] = 'B-' + list(label_types)[0]
                        word_id_labels = []
                    elif len(label_types) > 1:
                        raise ValueError(f"Conflicting labels for word_id {current_word_id} in id {id}: {word_id_labels}")
                    else:
                        # take majority label as fallback, using Counter
                        most_common_label = Counter(word_id_labels).most_common(1)[0][0]
                        new_bio_labels[current_word_id] = most_common_label
                        word_id_labels = []

                word_id_labels.append(label)
            # Word continues
            else:
                word_id_labels.append(label)
            
            current_word_id = row['word_id']

        new_records.append({
            'id': id,
            'model': model,
            'text': match_row['text'],
            'tokens': tokens,
            'ner_tags': match_row['ner_tags'],
            'pred_labels': new_bio_labels,
            'entities': match_row['entities'],
        })

    new_df = pd.DataFrame(new_records)
    # print length of dataframe
    print(len(new_df))
    new_file = file.replace('.csv', '_formatted.csv')
    new_df.to_csv(new_file, index=False, encoding='utf-8')

    return new_file
       
def get_all_ner_files(directory: str) -> list[str]:
    """Get all NER prediction files in a directory."""
    files = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv') and 'ner' in filename:
            if 'formatted' in filename:
                continue
            files.append(os.path.join(directory, filename))
    return files


def add_entities(file: str, tokens_col='tokens', bio_col='ner_tags', output_col='entities'):
    """ 
        Add an entities column to a data file based on the BIO-formatted NER tags.
        This is necessary to evaluate on entity level.
    """
    df = pd.read_csv(file)

    # add entities column
    df[output_col] = None
    for i, row in df.iterrows():
        ner_tags = ast.literal_eval(row[bio_col])
        tokens = ast.literal_eval(row[tokens_col])
        text = row['text']

        if pd.isna(text):
            df.at[i, output_col] = str([])
            continue

        entities = []
        cur_entity_tokens = []
        cur_entity_type = None

        # Step 1: collect entities tokens
        for token, label in zip(tokens, ner_tags):
            if label.startswith("B-"):
                if cur_entity_tokens:
                    entities.append((cur_entity_tokens, cur_entity_type))
                    cur_entity_tokens = []
                cur_entity_type = label[2:].lower().replace(" ", "-")
                cur_entity_tokens.append(token)

            elif label.startswith("I-") and cur_entity_type is not None:
                cur_entity_tokens.append(token)
            else:
                if cur_entity_tokens:
                    entities.append((cur_entity_tokens, cur_entity_type))
                    cur_entity_tokens = []
                    cur_entity_type = None

        # Step 2: convert entity tokens to strings
        entity_strings = []
        for entity_tokens, entity_type in entities:
            entity_string = get_entity_string_from_bio(
                entity_tokens, text)
            if entity_string:
                entity_strings.append((entity_string, entity_type))
            else:
                print(
                    f"Warning: Empty entity string for tokens {entity_tokens} in text '{text}'")

        df.at[i, output_col] = str(entity_strings)
    df.to_csv(file, index=False, encoding='utf-8')


def main():

    # Get all NER prediction files in a directory
    ner_files = get_all_ner_files(
        '/home/vera/Documents/Uni/Master/Master_Thesis2.0/PsyNamic-Scale/bert_baseline/predictions')

    for file in ner_files:
        print(f"Processing file: {file}")
        new_file = convert_berttoken_to_bio(
            file,
            '/home/vera/Documents/Uni/Master/Master_Thesis2.0/PsyNamic-Scale/zero_shot/ner/ner_gpt-4o-2024-08-06_22-07-22.csv'
        )
        add_entities(new_file, tokens_col='tokens', bio_col='pred_labels', output_col='pred_entities')


if __name__ == "__main__":
    main()
