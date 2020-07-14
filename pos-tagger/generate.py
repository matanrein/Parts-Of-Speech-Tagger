import csv
from io import open
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from constants import HMM_PICKLE_FILE, GENERATED_TAGS_FILE
import pickle
import argparse
import sys


def generate_tags(csv_file):
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
    words = []
    with open(csv_file, 'rt') as text_file:
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
    return tags_by_sentence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='this command evaluates the model trained by the train command')
    parser.add_argument('text_csv_file')
    args = parser.parse_args(sys.argv[1:])
    csv_text_file = args.text_csv_file
    tags_by_sentence = generate_tags(csv_text_file)
    with open(GENERATED_TAGS_FILE, 'wt') as tags_file:
        writer = csv.writer(tags_file)
        writer.writerows(tags_by_sentence)
    print('tags were written to ' + GENERATED_TAGS_FILE)
