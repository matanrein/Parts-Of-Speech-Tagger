import unittest
from preprocessing import build_ordinal_encoder, build_one_hot_encoder
import numpy as np


class EncodersTest(unittest.TestCase):
    def test_feature_one_hot_encoder(self):
        tags_train = np.array(['ADJ', 'NOUN', 'CCONJ', 'ADJ', 'NOUN', 'PUNCT', 'NOUN', 'ADP', 'NOUN']).reshape(-1, 1)
        tags_dev = np.array(['PROPN', 'PROPN', 'PROPN', 'PROPN', 'ADP', 'PROPN', 'PUNCT']).reshape(-1, 1)

        tags_to_test = np.array(['ADJ', 'NOUN']).reshape(-1, 1)

        encoder = build_one_hot_encoder(tags_train, tags_dev)
        assert (tags_train == encoder.inverse_transform(encoder.transform(tags_train))).all()
        assert (tags_to_test == encoder.inverse_transform(encoder.transform(tags_to_test))).all()

    def test_tag_encoder(self):
        words_train = np.array(
            ['Aesthetic', 'Appreciation', 'and', 'Spanish', 'Art', ':', 'Insights', 'from', 'Eye-Tracking']).reshape(-1,
                                                                                                                     1)
        words_dev = np.array(
            ['Claire', 'Bailey-Ross', 'claire.bailey-ross@port.ac.uk', 'University', 'of', 'Portsmouth', ',']).reshape(
            -1, 1)

        words_to_test = np.array(['Art', 'of']).reshape(-1, 1)

        encoder = build_ordinal_encoder(words_train, words_dev)
        assert (words_train == encoder.inverse_transform(encoder.transform(words_train))).all()
        assert (words_to_test == encoder.inverse_transform(encoder.transform(words_to_test))).all()
