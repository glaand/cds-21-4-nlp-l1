import os
import json
import openai
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
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        self.model = openai.ChatCompletion
        self.model_lang = "de"
        self.model_numb = 1

    def get_context(self, text):
        context = f"""
        I will give you a list of sentences separated by a korean won symbol (₩) and you have to tell me if it is positive, negative or neutral. 
        Please only return one of the three options per sentence. 
        If you are not sure, please return neutral.
        Please return the answer as a json list of strings.
        """
        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": '₩'.join(text)},
        ]
        return messages

    def predict(self, text):
        response = self.model.create(model="gpt-3.5-turbo", messages=self.get_context(text))
        predictions_group = response['choices'][0]['message']['content']
        predictions_group = json.loads(predictions_group)

        # check if is string
        if isinstance(predictions_group, str):
            predictions_group = [predictions_group]

        sentiments = []
        for prediction in predictions_group:
            sentiment = {"pos": 0.0, "neg": 0.0, "neut": 0.0}
            # lowercase and remove punctuation
            prediction = prediction.lower().replace(".", "").replace(",", "").replace("!", "").replace("?", "")
            if prediction == "negative":
                sentiment["neg"] = 1.0
            elif prediction == "neutral":
                sentiment["neut"] = 1.0
            elif prediction == "positive":
                sentiment["pos"] = 1.0
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
        self.data = pd.read_csv(self.filepath + f"/../../../facebook_dataset/{self.languages[language]}.csv")

    def run(self):
        self.load_data(self.model_lang)
        groups = []
        for group in tqdm(self.chunker(self.data['sentence_text'].to_numpy(), 100)):
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