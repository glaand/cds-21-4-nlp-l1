'''
Sources:
https://www.mdpi.com/2079-9292/9/3/483
https://www.mdpi.com/2071-1050/15/3/2573
https://www.researchgate.net/publication/350798422_Sentiment_Analysis_Of_Twitter_Data_By_Using_Deep_Learning_And_Machine_Learning
'''
import pandas as pd
import numpy as np
from tqdm import tqdm
import pathlib
from flair.models import TextClassifier
from flair.data import Sentence

class AbstractSentimentAnalysisModel:
    languages = {
        "de": "german",
        "en": "english",
    }
    filepath = str(pathlib.Path(__file__).parent.absolute())

    ### START CHANGING HERE
    def __init__(self):
        self.model = TextClassifier.load('en-sentiment')
        self.model_lang = "en"
        self.model_numb = 4

    def predict(self, text):
        predictions = []
        for sentence in text:
            obj = Sentence(sentence)
            self.model.predict(obj)
            label = obj.labels[0]
            pos, neg, neut = 0, 0, 0
            if label.value.lower() == 'positive':
                pos = label.score
            elif label.value.lower() == 'negative':
                neg = label.score
            else:
                neut = label.score
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
            for sentiment in group[1]:
                sentence_id, pos, neg, neut = sentences[curIndex][0], sentiment
                results.append((sentence_id, pos, neg, neut))
                curIndex += 1
        return results
    ### STOP CHANGING HERE

    def chunker(self, seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))

    def load_data(self, language):
        self.data = pd.read_csv(self.filepath + f"/../../../facebook_dataset/{self.languages[language]}.csv")

    def run(self):
        self.load_data(self.model_lang)
        groups = []
        for group in tqdm(self.chunker(self.data['sentence_text'].to_numpy(), 1000)):
            sentiments = []
            for text in group:
                score, label = self.predict(text)
                sentiments.append((score, label))
            groups.append((group, sentiments))
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