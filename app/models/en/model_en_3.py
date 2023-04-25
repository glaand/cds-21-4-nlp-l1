'''
Sources:
https://www.mdpi.com/2071-1050/11/16/4459
https://www.researchgate.net/publication/50378498_A_new_ANEW_Evaluation_of_a_word_list_for_sentiment_analysis_inmicroblogs
'''
import pandas as pd
import numpy as np
from tqdm import tqdm
import pathlib
from afinn import Afinn

class AbstractSentimentAnalysisModel:
    languages = {
        "de": "german",
        "en": "english",
    }
    filepath = str(pathlib.Path(__file__).parent.absolute())

    def __init__(self):
        self.model = Afinn()
        self.model_lang = "en"
        self.model_numb = 3

    def predict(self, text):
        predictions = []
        for sentence in text:
            prediction = self.model.score(sentence)
            pos, neg, neut = 0, 0, 0
            if prediction > 0:
                pos = prediction/5.0
                neut = 1-pos
            elif prediction < 0:
                neg = abs(prediction)/5.0
                neut = 1-neg
            else:
                neut = 1.0
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
                pos, neut, neg = sentiment
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
