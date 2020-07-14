import unittest
from load_data import get_data_from_conll
import itertools


class LoadDataTests(unittest.TestCase):
    def test_get_data_from_conll(self):
        feature_list, tag_list, sentence_lengths = get_data_from_conll('pos-tagger/tests/test-data/test.conllu')
        sentence1 = ['the', 'prevalence', 'of', 'discrimination', 'across', 'racial', 'group', 'in', 'contemporary',
                     'America', ':']
        sentence2 = ['result', 'from', 'a', 'nationally', 'representative', 'sample', 'of', 'adult']
        sentence3 = ['introduction', '.']

        tags1 = ['DET', 'NOUN', 'ADP', 'NOUN', 'ADP', 'ADJ', 'NOUN', 'ADP', 'ADJ', 'PROPN', 'PUNCT']
        tags2 = ['NOUN', 'ADP', 'DET', 'ADV', 'ADJ', 'NOUN', 'ADP', 'NOUN']
        tags3 = ['NOUN', 'PUNCT']

        real_sentence_lengths = [len(sentence1), len(sentence2), len(sentence3)]
        self.assertEqual(list(feature_list.reshape(feature_list.shape[0])),
                         list(itertools.chain(sentence1, sentence2, sentence3)))
        self.assertEqual(list(tag_list.reshape(tag_list.shape[0])), list(itertools.chain(tags1, tags2, tags3)))
        self.assertEqual(sentence_lengths, real_sentence_lengths)
