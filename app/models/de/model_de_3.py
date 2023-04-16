'''
Sources:
This German VADER model is based on the English VADER model and uses the same lexicon.
https://github.com/KarstenAMF/GerVADER.git
'''
import pandas as pd
import numpy as np
from tqdm import tqdm
import pathlib
import vaderSentiment
import vaderSentimentGer as GerVader
