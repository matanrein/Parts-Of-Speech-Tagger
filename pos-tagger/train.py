from load_data import get_data_from_conll
from preprocessing import build_one_hot_encoder, build_ordinal_encoder
from seqlearn.hmm import MultinomialHMM
from sklearn.metrics import accuracy_score
import numpy as np
import itertools
import scipy
import pickle
from io import open
from constants import HMM_PICKLE_FILE

train_corpus_file = '/home/matan/POS-Tagger/UD_English-GUM/en_gum-ud-train.conllu'
dev_corpus_file = '/home/matan/POS-Tagger/UD_English-GUM/en_gum-ud-dev.conllu'

train_features, train_tags, train_sentence_lengths = get_data_from_conll(train_corpus_file)
dev_features, dev_tags, dev_sentence_lengths = get_data_from_conll(dev_corpus_file)

feature_encoder = build_one_hot_encoder(train_features, dev_features)
tag_encoder = build_ordinal_encoder(train_tags, dev_tags)

train_feature_matrix = feature_encoder.transform(train_features)
dev_feature_matrix = feature_encoder.transform(dev_features)
train_tag_matrix = tag_encoder.transform(train_tags)
dev_tag_matrix = tag_encoder.transform(dev_tags)

X_train = train_feature_matrix
X_dev = dev_feature_matrix
Y_train = train_tag_matrix
Y_dev = dev_tag_matrix

hmm = MultinomialHMM()
hmm.fit(X_train, Y_train, train_sentence_lengths)

Y_dev_pred = hmm.predict(X_dev, dev_sentence_lengths)
print("Dev Accuracy: %.3f" % (100 * accuracy_score(Y_dev, Y_dev_pred)))

Y_train_pred = hmm.predict(X_train, train_sentence_lengths)
print("Train Accuracy: %.3f" % (100 * accuracy_score(Y_train, Y_train_pred)))
X_train_full = scipy.sparse.vstack([X_train, X_dev])
Y_train_full = np.concatenate([Y_train, Y_dev])
sentence_full_lengths = list(itertools.chain(train_sentence_lengths, dev_sentence_lengths))


print('Training on full dataset...')
hmm.fit(X_train_full, Y_train_full, sentence_full_lengths)
print('Saving HMM Model to ' + HMM_PICKLE_FILE)
pickle.dump((hmm, feature_encoder, tag_encoder), open(HMM_PICKLE_FILE, 'wb'))
hmm, feature_encoder, tag_encoder = pickle.load(open(HMM_PICKLE_FILE, 'rb'))