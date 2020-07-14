from conllu import parse
import itertools
import numpy as np

CONLL_WORD_FIELD = 'lemma'
#CONLL_WORD_FIELD = 'form'
CONLL_TAG_FIELD = 'upos'


def load_corpus(file):
    data_file = open(file, "r")
    corpus = parse(data_file.read()) # for large corpus that doesn't fit in memory parse_incr should be used
    data_file.close()
    return corpus


def data_gen_dict(corpus):
    for tokenlist in corpus:
        feature_list = []
        tag_list = []
        for token in tokenlist:
            feature_dict = {CONLL_WORD_FIELD: token[CONLL_WORD_FIELD]}
            tag_dict = {CONLL_TAG_FIELD: token[CONLL_TAG_FIELD]}
            feature_list.append(feature_dict)
            tag_list.append(tag_dict)
        yield feature_list, tag_list

def get_data_from_conll(corpus_file):
    corpus = load_corpus(corpus_file)
    data = data_gen_dict(corpus)
    sentences = [(sentence, tags) for (sentence, tags) in data]
    features, tags = list(zip(*sentences))
    feature_list = list(itertools.chain(*features))
    feature_list = list(map(lambda x: x[CONLL_WORD_FIELD], feature_list))
    feature_list = np.array(feature_list).reshape(-1, 1)
    tag_list = list(itertools.chain(*tags))
    tag_list = list(map(lambda x: x[CONLL_TAG_FIELD], tag_list))
    tag_list = np.array(tag_list).reshape(-1, 1)
    sentence_lengths = list(map(lambda x: len(x[0]), sentences))
    return feature_list, tag_list, sentence_lengths

