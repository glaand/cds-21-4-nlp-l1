FROM python:3.11.3-bullseye

WORKDIR /app
COPY requirements.txt /app
RUN pip install -r requirements.txt
RUN pip install pyicu
RUN polyglot download LANG:de
RUN python -m spacy download de_core_news_md
RUN python -c "import nltk; nltk.download('punkt')"
RUN python -c "import nltk; nltk.download('vader_lexicon')"
RUN python -c "from flair.models import TextClassifier; TextClassifier.load('en-sentiment')"
COPY . /app/
EXPOSE 8000
CMD ["bash", "start_chatbot.sh"]
