from germansentiment import SentimentModel
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
    ### START CHANGING HERE
    def __init__(self):
        self.model = SentimentModel()
        self.model_lang = "de"
        self.model_numb = 1

    def predict(self, text):
        return self.model.predict_sentiment(text, output_probabilities=True)
    
    def process_predictions(self, groups):
        results = []
        for group in tqdm(groups):
            for sentiment in group[1]:
                sentiments = {}
                for s in sentiment:
                    if s[0] == "negative":
                        sentiments["neg"] = s[1]
                    elif s[0] == "neutral":
                        sentiments["neut"] = s[1]
                    elif s[0] == "positive":
                        sentiments["pos"] = s[1]
                sentence_id, pos, neg, neut = self.data.to_numpy()[curIndex][0], sentiments["pos"], sentiments["neg"], sentiments["neut"]
                results.append((sentence_id, pos, neg, neut))
                curIndex += 1
        return results
    ### STOP CHANGING HERE



    def chunker(self, seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))
    
    def load_data(self, language):
        self.data = pd.read_csv(self.filepath + f"/../../facebook_dataset/{self.languages[language]}.csv")

    def run(self):
        self.load_data(self.model_lang)
        groups = []
        for group in tqdm(self.chunker(self.data['sentence_text'].to_numpy(), 1000)):
            groups.append(self.predict(group))
        results = self.process_predictions(groups)
        self.save_results(results)

    def save_results(self, results):
        results_df = pd.DataFrame(results, columns=['sentence_id', 'pos', 'neg', 'neut'])
        results_df.to_csv(self.filepath + f"/results/model_{self.model_lang}_{self.model_numb}.csv", index=False)

AbstractSentimentAnalysisModel().run()