'''
It defines a class named AbstractSentimentAnalysisModel that performs sentiment analysis on text data using the SentimentIntensityAnalyzer class from the nltk.sentiment module. The class has methods for loading data, chunking data, predicting sentiment, processing predictions, and saving results.

The run() method of the class calls load_data() to load data from a CSV file, and then calls chunker() and predict() to process the data in chunks of 1000 sentences. The predicted sentiment scores are then passed to process_predictions() to be processed and saved to a new CSV file using save_results().

The AbstractSentimentAnalysisModel class has two class attributes: languages and filepath, which define the supported languages for sentiment analysis and the path to the current file, respectively. The class also has three instance attributes: model, model_lang, and model_numb, which respectively hold an instance of SentimentIntensityAnalyzer, the language being used for sentiment analysis, and the model number.
'''
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
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
            nltk.data.find('sentiments/vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
        self.model = nltk.sentiment.SentimentIntensityAnalyzer()
        self.model_lang = "en"
        self.model_numb = 1

    def predict(self, text):
        predictions = []
        for sentence in text:
            prediction = self.model.polarity_scores(sentence)
            predictions.append({
                "pos": prediction['pos'],
                "neg": prediction['neg'],
                "neut": prediction['neu']
            })
        return predictions

    def process_predictions(self, groups):
        results = []
        sentences = self.data.to_numpy()
        curIndex = 0
        for group in tqdm(groups):
            for sentiment in group:
                sentence_id, pos, neg, neut = sentences[curIndex][0], sentiment['pos'], sentiment['neg'], sentiment['neut']
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