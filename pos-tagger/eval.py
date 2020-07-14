from constants import HMM_PICKLE_FILE
import pickle
from load_data import get_data_from_conll
from sklearn.metrics import accuracy_score
from io import open
import argparse
import sys


def eval(test_corpus_file):
    hmm, feature_encoder, tag_encoder = pickle.load(open(HMM_PICKLE_FILE, 'rb'))
    test_features, test_tags, test_sentence_lengths = get_data_from_conll(test_corpus_file)
    X_test = feature_encoder.transform(test_features)
    Y_test = tag_encoder.transform(test_tags)

    Y_test_pred = hmm.predict(X_test, test_sentence_lengths)
    print("Test Accuracy: %.3f" % (100 * accuracy_score(Y_test, Y_test_pred)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='this command evaluates the model trained by the train command')
    parser.add_argument('test_file')
    args = parser.parse_args(sys.argv[1:])
    eval(args.test_file)
