import pandas as pd
import numpy as np
import ctypes
import multiprocessing
import time
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from translations import translate_emoticons, translate_to_english, translate_to_german
# shared array for storing results
manager = multiprocessing.Manager()
results = manager.list([0] * 10)

def parallel_task(task):
    model_id, language, translated_sentence = task
    print(f"Running model {model_id} for language {language}...")
    start = time.time()
    module = __import__(f"models.{language}.model_{language}_{model_id}", fromlist=["AbstractSentimentAnalysisModel"])
    model = getattr(module, "AbstractSentimentAnalysisModel")
    i = model_id - 1    
    if language == "de":
        i += 5
    predicted = model().predict(translated_sentence)
    #try:
    #    predicted = model().predict(translated_sentence)
    #except Exception as e:
    #    predicted = [{"pos": 0.0, "neg": 0.0, "neut": 0.0} for _ in translated_sentence]
    end = time.time()
    print(f"Model {model_id} for language {language} finished in {end - start} seconds.")
    results[i] = pd.DataFrame(predicted, columns=["pos", "neg", "neut"])

class Pipeline:
    def translate_emoticons(self, sentence):
        print("Translating emoticons...")
        preprocessed_sentence = translate_emoticons(sentence)
        return preprocessed_sentence
    
    def translate_to_english(self, preprocessed_sentence):
        print("Translating to english...")
        translated_sentence = translate_to_english(preprocessed_sentence)
        print(f'Translated Sentence: {translated_sentence}')
        return translated_sentence
    
    def translate_to_german(self, preprocessed_sentence):
        print("Translating to german...")
        translated_sentence = translate_to_german(preprocessed_sentence)
        print(f'Translated Sentence: {translated_sentence}')
        return translated_sentence
    
    def sentiment_analysis_for_english(self, translated_sentence):
        print("Sentiment analysis for english...")
        tasks = [
            (1, "en", translated_sentence),
            (2, "en", translated_sentence),
            (3, "en", translated_sentence),
            (4, "en", translated_sentence),
            (5, "en", translated_sentence)
        ]
        return tasks
    
    def sentiment_analysis_for_german(self, translated_sentence):
        print("Sentiment analysis for german...")
        tasks = [
            (1, "de", translated_sentence),
            (2, "de", translated_sentence),
            (3, "de", translated_sentence),
            (4, "de", translated_sentence),
            (5, "de", translated_sentence)
        ]
        return tasks
    
    def run_parallel_tasks(self, english_tasks, german_tasks):
        print("Running parallel tasks...")
        tasks = english_tasks + german_tasks
        with multiprocessing.Pool() as pool:
            pool.map(parallel_task, tasks)
        return results[:5], results[5:]
    
    def count_simulated_participants_choices(self, sentiment_pairs):
        print("Counting simulated participants choices...")
        processed_dfs = []
        for english_sentiment, german_sentiment in sentiment_pairs:
            # check if variable is a dataframe
            if not isinstance(english_sentiment, pd.DataFrame):
                continue
            if not isinstance(german_sentiment, pd.DataFrame):
                continue
            # make pos, neut, neg columns positive
            english_sentiment["pos"] = english_sentiment["pos"].abs()
            english_sentiment["neut"] = english_sentiment["neut"].abs()
            english_sentiment["neg"] = english_sentiment["neg"].abs()
            german_sentiment["pos"] = german_sentiment["pos"].abs()
            german_sentiment["neut"] = german_sentiment["neut"].abs()
            german_sentiment["neg"] = german_sentiment["neg"].abs()

            merged_df = pd.concat([english_sentiment, german_sentiment], axis=0, ignore_index=True)
            if "sentence_id" not in merged_df.columns:
                df = merged_df[["pos", "neut", "neg"]].mean()
                df = pd.DataFrame(df).T
                df["max_prob"] = df[["pos", "neut", "neg"]].max(axis=1)
            else:
                df = merged_df.groupby("sentence_id", as_index=False)[["pos", "neut", "neg"]].mean()
                df["max_prob"] = df[["pos", "neut", "neg"]].max(axis=1)

            # convert probabilities to 0 or 1 with max prob as threshold
            df.loc[df["pos"] < df["max_prob"], "pos"] = 0
            df.loc[df["pos"] == df["max_prob"], "pos"] = 1
            df.loc[df["neut"] < df["max_prob"], "neut"] = 0
            df.loc[df["neut"] == df["max_prob"], "neut"] = 1
            df.loc[df["neg"] < df["max_prob"], "neg"] = 0
            df.loc[df["neg"] == df["max_prob"], "neg"] = 1
            df = df.drop("max_prob", axis=1)

            # convert pos,neut,neg to int
            df["pos"] = df["pos"].astype(int)
            df["neut"] = df["neut"].astype(int)
            df["neg"] = df["neg"].astype(int)
            processed_dfs.append(df)

        merged_df = pd.concat(processed_dfs, axis=0, ignore_index=True)
        if "sentence_id" not in merged_df.columns:
            counted_df = merged_df[["pos", "neut", "neg"]].sum()
            counted_df = pd.DataFrame(counted_df).T
        else:
            counted_df = merged_df.groupby("sentence_id", as_index=False)[["pos", "neut", "neg"]].sum()

        # replace 2 with 1 for pos, neut, neg
        counted_df.loc[counted_df["pos"] == 2, "pos"] = 1
        counted_df.loc[counted_df["neut"] == 2, "neut"] = 1
        counted_df.loc[counted_df["neg"] == 2, "neg"] = 1

        return counted_df
    
    def run(self, sentence):
        preprocessed_sentence = self.translate_emoticons(sentence)
        english_sentence = self.translate_to_english(preprocessed_sentence)
        german_sentence = self.translate_to_german(english_sentence)    # EASIER TO TRANSLATE FROM ENGLISH TO STANDARD GERMAN
        english_sentiments_tasks = self.sentiment_analysis_for_english(english_sentence)
        german_sentiments_tasks = self.sentiment_analysis_for_german(german_sentence)
        english_sentiments, german_sentiments = self.run_parallel_tasks(english_sentiments_tasks, german_sentiments_tasks)
        sentiment_pairs = list(zip(english_sentiments, german_sentiments))
        experiment = self.count_simulated_participants_choices(sentiment_pairs)
        return experiment, results