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
        predictions_group = self.model.predict_sentiment(text, output_probabilities=True)
        sentiments = []
        for predictions in predictions_group[1]:
            sentiment = {}
            for prediction in predictions:
                if prediction[0] == "negative":
                    sentiment["neg"] = prediction[1]
                elif prediction[0] == "neutral":
                    sentiment["neut"] = prediction[1]
                elif prediction[0] == "positive":
                    sentiment["pos"] = prediction[1]
            sentiments.append(sentiment)
        return sentiments
    
    def process_predictions(self, groups):
        results = []
        sentences = self.data.to_numpy()
        curIndex = 0
        for group in tqdm(groups):
            for sentiment in group:
                sentence_id, pos, neg, neut = sentences[curIndex][0], sentiment["pos"], sentiment["neg"], sentiment["neut"]
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
        results_df['sentence_id'] = results_df['sentence_id'].astype(int)
        results_df['pos'] = results_df['pos'].apply(lambda x: np.round(x, 5))
        results_df['neg'] = results_df['neg'].apply(lambda x: np.round(x, 5))
        results_df['neut'] = results_df['neut'].apply(lambda x: np.round(x, 5))
        results_df.to_csv(self.filepath + f"/results/model_{self.model_lang}_{self.model_numb}.csv", index=False)

if __name__ == "__main__":
    AbstractSentimentAnalysisModel().run()