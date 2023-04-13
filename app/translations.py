from googletrans import Translator
import re
from emot import emot

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
    text = text.pop()
    emotions = emot().emoticons(text)
    correction = 0
    for i, location in enumerate(emotions['location']):
        emoticon = emotions['value'][i]
        start = location[0] + correction
        end = location[1] + correction
        meaning = emotions['mean'][i]
        replacement = replacer(meaning)
        text = text[:start] + replacement + text[end:]
        correction += len(replacement) - len(emoticon)     # correction for the length of the emoticon
    return [f'{text}']

def translate_to_english(text):
    translator = Translator()
    translate = text[0]
    result = translator.translate(translate, src='de', dest='en').text
    print(f'Type: {type(translate)}. Length: {len(translate)}. Result: {result}')
    return [f'{result}']

def translate_to_german(text):
    translator = Translator()
    translate = text[0]
    result = translator.translate(translate, src='en', dest='de').text
    print(f'Type: {type(translate)}. Length: {len(translate)}. Result: {result}')
    return [f'{result}']