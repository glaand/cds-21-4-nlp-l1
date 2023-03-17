# https://github.com/noe-eva/NOAH-Corpus
# https://huggingface.co/noeminaepli/swiss_german_pos_model

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

model = AutoModelForTokenClassification.from_pretrained("noeminaepli/swiss_german_pos_model")
tokenizer = AutoTokenizer.from_pretrained("noeminaepli/swiss_german_pos_model")

pos_tagger = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")
tokens = pos_tagger("Hesch du eh idee wie ma das chönnti mache?")
print(tokens)