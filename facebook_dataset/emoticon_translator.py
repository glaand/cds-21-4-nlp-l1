from emot.core import emot
from googletrans import Translator  # version 3.1.0a0 use: pip install googletrans==3.1.0a0
import re
import pandas as pd
from tqdm import tqdm
FACEBOOK_PATH = "cds-21-4-nlp-l1/facebook_dataset/facebook.full.csv"
CHATMANIA_PATH = "cds-21-4-nlp-l1/dataset/chatmania.csv"

def replacer(meanings):
    regex = re.compile(r'^[^\s,]+,')
    match = regex.match(meanings)
    if match:
        return match.group(0)[:-1]
    
    regex = re.compile(r'^[^,]+,\s*(\w+)')
    match = regex.match(meanings)
    if match:
        return match.group(1)
    
    regex = re.compile(r'^\s*([\w\s]+?)\s*or\b')
    match = regex.match(meanings)
    if match:
        return match.group(1)
    
    return meanings

def translate_emoticons(text):
    emotions = emot().emoticons(text)
    correction = 0
    for i, location in enumerate(emotions['location']):
        emoticon = emotions['value'][i]
        start = location[0] + correction
        end = location[1] + correction
        meaning = emotions['mean'][i]
        replacement = replacer(meaning)
        text = text[:start] + replacement + text[end:]
        correction += len(replacement) - len(emoticon)
    return text

if __name__ == "__main__":
    # load data
    df_facebook = pd.read_csv(FACEBOOK_PATH)
    df_facebook = df_facebook[['sentence_id', 'sentence_text']]
    df_chatmania = pd.read_csv(CHATMANIA_PATH)
    # merge data
    merged_df = pd.concat([df_facebook, df_chatmania], ignore_index=True)
    # translate emoticons
    tqdm.pandas()
    for i, row in tqdm(merged_df.iterrows(), total=merged_df.shape[0]):
        text = row['sentence_text']
        merged_df.at[i, 'sentence_text'] = translate_emoticons(text)
    # translate to english & german
    english_df = merged_df.copy().assign(translate_success=False)
    german_df = merged_df.copy().assign(translate_success=False)
    translator = Translator()
    failed_translations_indexes = []
    for i, row in tqdm(merged_df.iterrows(), total=merged_df.shape[0]):
        try:
            translated_engl = translator.translate(row['sentence_text'], dest='en').text
            english_df.loc[i] = [row['sentence_id'], translated_engl, True]
            translated_ger = translator.translate(translated_engl, dest='de').text
            german_df.loc[i] = [row['sentence_id'], translated_ger, True]
        except:
            failed_translations_indexes.append(i)
            english_df.loc[i] = [row['sentence_id'], row['sentence_text'], False]
            german_df.loc[i] = [row['sentence_id'], row['sentence_text'], False]
            continue
    print(f"Failed translations: {len(failed_translations_indexes)}")
    # retry failed translations until success
    while failed_translations_indexes:
        last_index = failed_translations_indexes.pop()
        # try translating the sentence again
        row = merged_df.loc[last_index]
        try:
            translated_engl = translator.translate(row['sentence_text'], dest='en').text
            english_df.loc[i] = [row['sentence_id'], translated_engl, True]
            translated_ger = translator.translate(translated_engl, dest='de').text
            german_df.loc[i] = [row['sentence_id'], translated_ger, True]
        except:
            print(f"Failed to translate: Row {last_index}")
            failed_translations_indexes.append(last_index)
            continue
    english_df.to_csv("facebook_dataset/english.csv", index=False)
    german_df.to_csv("facebook_dataset/german.csv", index=False)
        