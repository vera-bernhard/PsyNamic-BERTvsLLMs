import pandas as pd
import ast
from evaluation.parsing import get_entity_string_from_bio


def add_entities(file: str):
    """ 
        Add an entities column to a data file based on the BIO-formatted NER tags.
        This is necessary to evaluate on entity level.
    """
    df = pd.read_csv(file)

    # add entities column
    df['entities'] = None
    for i, row in df.iterrows():
        ner_tags = ast.literal_eval(row['ner_tags'])
        tokens = ast.literal_eval(row['tokens'])
        text = row['text']

        if pd.isna(text):
            df.at[i, 'entities'] = str([])
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

        df.at[i, 'entities'] = str(entity_strings)
    df.to_csv(file, index=False, encoding='utf-8')
