import os

import numpy as np
import pandas as pd
import re, gensim
import logging

from models.preprocess import get_data
from models.stopwords import Utils, remove_stopwords

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# sciket-learn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.model_selection import GridSearchCV
from pprint import pprint

# Plotting tools
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt

# load stopword
stopwords = Utils().load_stopwords()

# data path
data_path = os.path.dirname(os.path.realpath(os.getcwd()))
data_path = os.path.join(data_path, 'data')
data_train = os.path.join(data_path, 'data_test', 'test_test')

# Read data
X_train, y_train = get_data(data_train)

# Remove stopword
docs = remove_stopwords(X_train, stopwords)

# model
lda_model = LatentDirichletAllocation(n_components=10, max_iter=200, learning_method='online', random_state=100,
                                      batch_size=200, evaluate_every=-1, n_jobs=-1)
lda_model_out = lda_model.fit_transform(docs)
print(lda_model)
