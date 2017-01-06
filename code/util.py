from prefixspan import prefixspan
import pandas as pd
import numpy as np

def match_sequences(encoded_sentence, sequences):
    pfs = prefixspan([encoded_sentence])
    pfs.mining_sequence({},0)
    result = pfs.parse_results_unsorted()
    target = [x[0] for x in sequences]
    features = []
    length = len(encoded_sentence)
    for sequence in target:
        if sequence in result.keys():
            features.append(result[sequence] / length)
        else:
            features.append(0)
    return features


def generate_train_test(data_df):
    # data_df = pd.read_csv('data/sentence_component.csv',header = 0)
    true = data_df.loc[data_df['Label'] == 1]
    false = data_df.loc[data_df['Label'] == 0]
    train_true,test_true = _split_data(true)
    train_false,test_false = _split_data(false)
    train = train_false.append(train_true)
    test = test_false.append(test_true)
    return train, test

def _split_data(data_df):
    msk = np.random.rand(len(data_df)) < 0.8
    train_data = data_df[msk]
    test_data = data_df[~msk]
    return train_data, test_data