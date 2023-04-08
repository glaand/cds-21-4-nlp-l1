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
        scoring, label = 0, 'pos'
        return scoring, label
    
    def sentiment_analysis_for_german(self, translated_sentence):
        scoring, label = 0, 'pos'
        return scoring, label
    
    def calculate_average_scoring_matching(self, english_sentiment, german_sentiment):
        average_scoring = 0
        # since sentiment machine learning not ready yet, we just return pos for development purposes for ajshe
        average_label = 'pos'
        return average_scoring, average_label
    
    def run(self, sentence):
        preprocessed_sentence = self.translate_emoticons(sentence)
        english_sentence = self.translate_to_english(preprocessed_sentence)
        german_sentence = self.translate_to_german(preprocessed_sentence)
        english_sentiment = self.sentiment_analysis_for_english(english_sentence)
        german_sentiment = self.sentiment_analysis_for_german(german_sentence)
        average_scoring, average_label = self.calculate_average_scoring_matching(english_sentiment, german_sentiment)
        return average_scoring, average_label 