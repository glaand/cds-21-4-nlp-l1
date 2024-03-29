'''
Sources:
https://pypi.org/project/textblob-de/
'''
from textblob_de import TextBlobDE as TextBlob
import nltk
import pandas as pd
import numpy as np
from tqdm import tqdm
import pathlib

class AbstractSentimentAnalysisModel:
    languages = {
        "de": "german",
        "en": "english",
    }
    filepath = str(pathlib.Path(__file__).parent.absolute())

    def __init__(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        self.model_lang = "de"
        self.model_numb = 2

    def predict(self, text):
        predictions = []
        for sentence in text:
            blob = TextBlob(sentence)
            sentiment = blob.sentiment.polarity
            if sentiment > 0:
                pos = sentiment
                neg = 1 - sentiment
                neut = 0
            elif sentiment < 0:
                pos = 1 - abs(sentiment)
                neg = abs(sentiment)
                neut = 0
            else:
                pos = 0
                neg = 0
                neut = 1
            predictions.append({
                "pos": pos,
                "neg": neg,
                "neut": neut
            })
        return predictions

    def process_predictions(self, groups):
        results = []
        sentences = self.data.to_numpy()
        curIndex = 0
        for group in tqdm(groups):
            for sentiment in group:
                sentence_id = sentences[curIndex][0]
                pos, neg, neut = sentiment
                results.append((sentence_id, pos, neg, neut))
                curIndex += 1
        return results

    def chunker(self, seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))

    def load_data(self, language):
        self.data = pd.read_csv(self.filepath + f"/../../../facebook_dataset/{self.languages[language]}.csv")

    def run(self):
        self.load_data(self.model_lang)
        groups = []
        for group in tqdm(self.chunker(self.data['sentence_text'].to_numpy(), 1000)):
            groups.append(self.predict(group))
        results = self.process_predictions(groups)
        self.save_results(results)

    def save_results(self, results):
        results_df = pd.DataFrame(results, columns=['sentence_id', 'pos', 'neg', 'neut'])
        results_df['sentence_id'] = results_df['sentence_id'].astype(int)
        results_df['pos'] = results_df['pos'].apply(lambda x: np.round(x, 5))
        results_df['neg'] = results_df['neg'].apply(lambda x: np.round(x, 5))
        results_df['neut'] = results_df['neut'].apply(lambda x: np.round(x, 5))
        results_df.to_csv(self.filepath + f"/results/model_{self.model_lang}_{self.model_numb}.csv", index=False)

if __name__ == "__main__":
    AbstractSentimentAnalysisModel().run()
