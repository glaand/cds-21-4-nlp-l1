FROM python:3.11.3-bullseye

WORKDIR /app
COPY requirements.txt /app
RUN pip install -r requirements.txt
RUN polyglot download LANG:de
RUN python -m spacy download de_core_news_md
COPY * /app/
EXPOSE 8000
CMD ["bash", "start_chatbot.sh"]
