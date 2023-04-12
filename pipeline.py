import pandas as pd
import numpy as np

class Pipeline:
    def translate_emoticons(self, sentence):
        preprocessed_sentence = sentence
        return preprocessed_sentence
    
    def translate_to_english(self, preprocessed_sentence):
        translated_sentence = preprocessed_sentence
        return translated_sentence
    
    def translate_to_german(self, preprocessed_sentence):
        translated_sentence = preprocessed_sentence
        return translated_sentence
    
    def sentiment_analysis_for_english(self, translated_sentence):
        english_sentiments = []
        return english_sentiments
    
    def sentiment_analysis_for_german(self, translated_sentence):
        german_sentiments = []
        return german_sentiments
    
    def count_simulated_participants_choices(self, sentiment_pairs):
        processed_dfs = []
        for english_sentiment, german_sentiment in sentiment_pairs:
            # make pos, neut, neg columns positive
            english_sentiment["pos"] = english_sentiment["pos"].abs()
            english_sentiment["neut"] = english_sentiment["neut"].abs()
            english_sentiment["neg"] = english_sentiment["neg"].abs()
            german_sentiment["pos"] = german_sentiment["pos"].abs()
            german_sentiment["neut"] = german_sentiment["neut"].abs()
            german_sentiment["neg"] = german_sentiment["neg"].abs()

            merged_df = pd.concat([english_sentiment, german_sentiment], axis=0, ignore_index=True)
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

            # handle multiple columns with 1
            df["ones_count"] = df[["pos", "neut", "neg"]].sum(axis=1)
            df.loc[df["ones_count"] >= 2, "neut"] = 0
            df.loc[df["ones_count"] >= 2, "pos"] = 0
            df.loc[df["ones_count"] >= 2, "neg"] = 0
            df = df.drop("ones_count", axis=1)
            df = df.groupby("sentence_id", as_index=False)[["pos", "neut", "neg"]].sum()

            # convert pos,neut,neg to int
            df["pos"] = df["pos"].astype(int)
            df["neut"] = df["neut"].astype(int)
            df["neg"] = df["neg"].astype(int)
            processed_dfs.append(df)

        merged_df = pd.concat(processed_dfs, axis=0, ignore_index=True)
        counted_df = merged_df.groupby("sentence_id", as_index=False)[["pos", "neut", "neg"]].sum()
        return counted_df
    
    def run(self, sentence):
        preprocessed_sentence = self.translate_emoticons(sentence)
        english_sentence = self.translate_to_english(preprocessed_sentence)
        german_sentence = self.translate_to_german(preprocessed_sentence)
        english_sentiments = self.sentiment_analysis_for_english(english_sentence)
        german_sentiments = self.sentiment_analysis_for_german(german_sentence)
        sentiment_pairs = list(zip(english_sentiments, german_sentiments))
        experiment = self.count_simulated_participants_choices(sentiment_pairs)
        return experiment 