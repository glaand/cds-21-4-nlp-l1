'''
Sources:
This model can be used for German and English. However, the sentiment analysis is only available for English. 
Source code is written in Java, can be used in Python via the CoreNLP wrapper.

https://github.com/stanfordnlp/CoreNLP
https://stanfordnlp.github.io/CoreNLP/other-languages.html#python
https://www.districtdatalabs.com/syntax-parsing-with-corenlp-and-nltk'''
import pandas as pd
import numpy as np
from tqdm import tqdm
import pathlib
import stanfordnlp

