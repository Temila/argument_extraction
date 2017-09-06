import re
import xgboost as xgb
import requests
import json
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from util import read_essay_txt_file, read_essay_ann_file, balance_data, generate_train_test_2
# from sequential_pattern import sequential_pattern
from sentence_component import Sentence_component
from sematic_similarity import Sematic_Similarity
from sklearn.neighbors import KNeighborsClassifier

# sp = sequential_pattern()
ss = Sematic_Similarity()
#load xgboost model
accuracy = []
recall = []
f1 = []

punctuation = '"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~'

def calculate_accuracy(detected_claims, claims):
    count = 0
    for detected_claim in detected_claims:
        for claim in claims:
            if claim in detected_claim or detected_claim in claim:
                count += 1
                break
    try:
        return float(count) / len(detected_claims)
    except:
        return 0.0

def calculate_recall(detected_claims, claims):
    count = 0
    for claim in claims:
        for detected_claim in detected_claims:
            if claim in detected_claim or detected_claim in claim:
                count += 1
                break
    return float(count) / len(claims)+0.000001

def generate_sub_sentence(sentence):
    min_size = 4
    r = re.compile(r'[{}]'.format(re.escape(punctuation)))
    chunck_by_punc = r.split(sentence)
    candidates = [x for x in chunck_by_punc if len(x.split(' ')) >= min_size]
    return candidates

def train_2():
    features = ['Mas','Ecs_1','Ecs_2','Ecs_3','Snp','Cnf_1','Cnf_2','Cnf_3','Cnf_4','Sub','Sen','Slc']
    label = ['Label']
    neighs = []
    data_df = pd.read_csv('New_data/csv/data_2.csv',header = 0)
    for i in range(1,28):
        train_df,test_df = generate_train_test_2(data_df,3000,i)
        data = train_df.append(test_df)
        neigh = KNeighborsClassifier(n_neighbors=47, algorithm='auto')
        neigh.fit(data[features],data[label])
        neighs.append(neigh)
    return neighs


def train():
    features = ['Mas','Ecs_1','Ecs_2','Ecs_3','Snp','Cnf_1','Cnf_2','Cnf_3','Cnf_4','Sub','Sen','Slc']
    label = ['Label']
    data_df = pd.read_csv('New_data/csv/data_2.csv',header = 0)
    train_df = balance_data(data_df,features,label)
    neigh = KNeighborsClassifier(n_neighbors=48, algorithm='auto')
    neigh.fit(train_df[features],train_df[label]) 
    return neigh

print 'training model'
bsts = train_2() 

for i in range(1,403):
    topic, data = read_essay_txt_file(i)
    claims = read_essay_ann_file(i)
    sentences = sent_tokenize(data)
    #encode sentence
    # headers = ['Mas','Ecs_1','Ecs_2','Ecs_3','Snp','Cnf_1','Cnf_2','Cnf_3','Cnf_4','Sub','Sen']
    headers = ['Mas','Ecs_1','Ecs_2','Ecs_3','Snp','Cnf_1','Cnf_2','Cnf_3','Cnf_4','Sub','Sen','Slc']
    all_setence_features = []
    print 'processing essay {}, {} sentences found'.format(str(i),len(sentences))
    index = 0
    count = 0
    match_list = {}
    for sentence in sentences:
        index += 1
        sub_sentences = generate_sub_sentence(sentence)
        for sub_sentence in sub_sentences:
            count += 1
            sc = Sentence_component(sentence,topic,ss)
            features = sc.extract_features()[1:]
            all_setence_features.append(features)
            match_list[str(count)] = index
        print 'sentence {} done'.format(index)
    # dtest = xgb.DMatrix(all_setence_features)
    nr = len(all_setence_features)
    preds = np.zeros(nr)
    for bst in bsts:
        p = bst.predict_proba(all_setence_features)
        l = np.array([ 1 if i[1] > i[0] else 0 for i in p])
        preds = np.add(preds,l)
    detected_claims_raw = []
    for i in range(len(preds)):
        if preds[i] > 14:
            detected_claims_raw.append(sentences[int(match_list[str(i+1)])-1])
    detected_claims = list(set(detected_claims_raw))
    # print detected_claims
    if len(detected_claims) == 0:
        max_index = np.argmax(preds)
        sent = sentences[int(match_list[str(max_index+1)])-1]
        # print sent
        detected_claims.append(sent)
    # print detected_claims
    a = calculate_accuracy(detected_claims,claims)
    r = calculate_recall(detected_claims,claims)
    if a == 0 and r == 0:
        f = 0
    else:
        f = 2*(a*r)/(a+r)
    print 'accuracy: {}, recall: {}, F1 score: {}'.format(str(a),str(r),str(f))
    # with open('New_data/result/test_with_essay/result_KNN_group.csv','a+') as file:
    #     file.write('{},{},{}\n'.format(str(a),str(r),str(f)))