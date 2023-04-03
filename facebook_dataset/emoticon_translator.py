from emot.core import emot
from googletrans import Translator  # version 3.1.0a0 use: pip install googletrans==3.1.0a0
import re
import pandas as pd
from tqdm import tqdm
FACEBOOK_PATH = "cds-21-4-nlp-l1/facebook_dataset/facebook.full.csv"
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
    translator = Translator()
    correction = 0
    for i, location in enumerate(emotions['location']):
        emoticon = emotions['value'][i]
        start = location[0] + correction
        end = location[1] + correction
        meaning = emotions['mean'][i]
        replacement = replacer(meaning)
        text = text[:start] + replacement + text[end:]
        correction += len(replacement) - len(emoticon)
    return translator.translate(translator.translate(text, src='de', dest='en').text, src='en', dest='de').text

if __name__ == "__main__":
    # load data
    df = pd.read_csv(FACEBOOK_PATH)
    tqdm.pandas()
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            df.at[i, 'sentence_text'] = translate_emoticons(row['sentence_text'])
        except:
            print(f"Error translating {i}th sentence.")
            i -= 1 # retry the current sentence
            continue
    df.to_csv('cds-21-4-nlp-l1/facebook_dataset/translated.csv', mode='a', index=False, header=False)
    print(f"Translated {len(df)} sentences")