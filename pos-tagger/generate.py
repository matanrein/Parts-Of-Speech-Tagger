import csv
from io import open
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
from constants import HMM_PICKLE_FILE
import pickle

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
# load to csv
CSV_FILE = 'pos-tagger/tests/test-data/test-sentences'
GENERATED_TAGS_FILE = 'generated-tags-file.txt'
words = []
with open(CSV_FILE, 'rt') as text_file:
    lines = list(csv.reader(text_file))
lemmatized = [lemmatizer.lemmatize(word) for line in lines for word in line]
lengths = [len(line) for line in lines]
X_test = np.array(lemmatized).reshape(-1,1)
hmm, feature_encoder, tag_encoder = pickle.load(open(HMM_PICKLE_FILE, 'rb'))
X_test = feature_encoder.transform(X_test)
Y_pred = hmm.predict(X_test, lengths).reshape(-1, 1)
pred_tags = tag_encoder.inverse_transform(Y_pred).reshape(-1)
print(pred_tags.shape)
split_indices = np.cumsum(lengths)
tags_by_sentence = np.split(pred_tags, split_indices)
with open(GENERATED_TAGS_FILE, 'wt') as tags_file:
    writer = csv.writer(tags_file)
    writer.writerows(tags_by_sentence)
print('tags were written to ' + GENERATED_TAGS_FILE)
