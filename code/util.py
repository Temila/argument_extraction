from prefixspan import prefixspan
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import string
import itertools
from imblearn.combine import SMOTETomek

def match_sequences(encoded_sentence, sequences):
    pfs = prefixspan([encoded_sentence],0)
    pfs.mining_sequence({},0)
    result = pfs.parse_results_unsorted()
    target = sequences
    features = []
    length = len(encoded_sentence)
    for sequence in target:
        if sequence in result.keys():
            features.append(1)
        else:
            features.append(0)
    return features


def generate_train_test(data_df):
    # data_df = pd.read_csv('data/sentence_component.csv',header = 0)
    print 'generating train test data'
    true = data_df.loc[data_df['Label'] == 1]
    false = data_df.loc[data_df['Label'] == 0]
    train_true,test_true = _split_data(true)
    train_false,test_false = _split_data(false)
    train = train_false.append(train_true)
    test = test_false.append(test_true)
    return train, test

def balance_data(data_df,features,label):
    sm = SMOTETomek(n_jobs=8)
    X = data_df[features]
    y = data_df[label]
    y = np.ravel(y)
    print 'resampling'
    X_resampled, y_resampled = sm.fit_sample(X,y)
    X_df = pd.DataFrame(X_resampled,columns = features)
    y_df = pd.DataFrame(y_resampled,columns = label)
    new_data_df = X_df.join(y_df)
    print 'Done'
    return new_data_df

def _split_data(data_df):
    msk = np.random.rand(len(data_df)) < 0.8
    train_data = data_df[msk]
    test_data = data_df[~msk]
    return train_data, test_data


def extract_candidate_chunks(text, grammar=r'KT:{(<JJ>*<NN.*>+<IN>)?<JJ>*<NN.*>+}'):
    # exclude candidates that are stop words or entirely punctuation
    punct = set(string.punctuation)
    stop_words = set(stopwords.words('english'))
    # tokenize, POS-tag, and chunk using regular expressions
    chunker = nltk.chunk.regexp.RegexpParser(grammar)
    tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text))
    all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent)) for tagged_sent in tagged_sents))
    # join constituent chunk words into a single chunked phrase
    candidates = [' '.join(word for word, pos, chunk in group) for key, group in itertools.groupby(all_chunks, lambda (word,pos,chunk): chunk != 'O') if key]

    return [cand for cand in candidates if cand not in stop_words and not all(char in punct for char in cand)]

def read_file(file, cut = False):
    with open(file,'r') as f:
        data = f.read().split('\n')
    result = []
    for d in data:
        if cut:
            result.append(d.split('<_break_>')[:-1])
        else:
            result.append(d.split('<_break_>'))
    return result

def read_essay_txt_file(file_number):
    file = 'ArgumentAnnotatedEssays/essay{}.txt'.format(str(file_number).zfill(3))
    with open(file,'r') as f:
        data = f.read()
    temp = data.split('\n\n')
    return temp[0], temp[1]

def read_essay_ann_file(file_number):
    file = 'ArgumentAnnotatedEssays/essay{}.ann'.format(str(file_number).zfill(3))
    with open(file,'r') as f:
        data = f.read()
    data = data.split('\n')
    claims = []
    for line in data:
        if line.startswith('T'):
            claims.append(line.split('\t')[2])
    return claims