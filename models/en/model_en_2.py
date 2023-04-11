'''
Sources:
https://www.researchgate.net/publication/348637239_Sentiment_Analysis_on_Twitter_by_Using_TextBlob_for_Natural_Language_Processing
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9910766/
'''
from textblob import TextBlob
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
        self.model_lang = "en"
        self.model_numb = 2

    def predict(self, text):
        return [TextBlob(sentence).sentiment.polarity for sentence in text] # I HAD TO CHANGE THIS AS THE SENTECE HAS TO BE CALLED IN THE FUNCTION

    def process_predictions(self, groups):
        results = []
        sentences = self.data.to_numpy()
        curIndex = 0
        for group in tqdm(groups):
            for sentiment in group:
                sentence_id, polarity = sentences[curIndex][0], sentiment
                results.append((sentence_id, polarity))
                curIndex += 1
        return results

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
        results_df = pd.DataFrame(results, columns=['sentence_id', 'polarity'])
        results_df['sentence_id'] = results_df['sentence_id'].astype(int)
        results_df['polarity'] = results_df['polarity'].apply(lambda x: np.round(x, 5))
        results_df.to_csv(self.filepath + f"/results/model_{self.model_lang}_{self.model_numb}.csv", index=False)

AbstractSentimentAnalysisModel().run()