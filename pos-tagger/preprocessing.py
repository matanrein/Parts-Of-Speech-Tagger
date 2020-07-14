from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import numpy as np


def build_one_hot_encoder(train_features, dev_features):
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
    features = np.concatenate([train_features, dev_features])
    one_hot_encoder.fit(features)
    return one_hot_encoder


def build_ordinal_encoder(train_tags, dev_tags):
    # for large datasets that don't fit in memory, FeatureHasher should be used in batches
    word_list = np.concatenate([train_tags, dev_tags])
    ordinal_encoder = OrdinalEncoder()
    ordinal_encoder.fit(word_list)
    return ordinal_encoder
