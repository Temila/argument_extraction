import re
import xgboost as xgb
import requests
import json
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from util import read_essay_txt_file, read_essay_ann_file, balance_data, generate_train_test_2
# from sequential_pattern import sequential_pattern
from sklearn.neighbors import KNeighborsClassifier

# sp = sequential_pattern()
#load xgboost model
accuracy = []
recall = []
f1 = []

punctuation = '"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~'

def calculate_percison(preds, labels):
    count = len(preds)
    a = 0.0
    b = 0.0
    for i in range(count):
        if preds[i] == 1:
            a += 1
            if labels[i] == 1:
                b += 1
    return float(b)/float(a)

def calculate_recall(preds, labels):
    count = len(preds)
    a = 0.0
    b = 0.0
    for i in range(count):
        if labels[i] == 1:
            a += 1
            if preds[i] == 1:
                b += 1
    return float(b)/float(a)

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
    return neighs, 14


def train():
    features = ['Mas','Ecs_1','Ecs_2','Ecs_3','Snp','Cnf_1','Cnf_2','Cnf_3','Cnf_4','Sub','Sen','Slc']
    label = ['Label']
    data_df = pd.read_csv('New_data/csv/data_2.csv',header = 0)
    train_df = balance_data(data_df,features,label)
    neigh = KNeighborsClassifier(n_neighbors=48, algorithm='auto')
    neigh.fit(train_df[features],train_df[label])
    return [neigh], 0

def read_sim():
    with open('New_data/sims_essay','r') as f:
        return f.read().split('\n')

print 'training model'
bsts, threshold = train()

data = pd.read_csv('New_data/csv/data_np.csv', sep="#", header = 0, index_col=False)
features = ['Mas','Ecs_1','Ecs_2','Ecs_3','Snp','Cnf_1','Cnf_2','Cnf_3','Cnf_4','Sub','Sen','Slc']
label = ['Label']
dtest = xgb.DMatrix(data[features],data[label])
labels = dtest.get_label()
nr = dtest.num_row()

preds = np.zeros(nr)
sims = read_sim()
for bst in bsts:
    p = bst.predict_proba(data[features])
    l = np.array([ 1 if i[1] > i[0] else 0 for i in p])
    preds = np.add(preds,l)

preds_norm = np.zeros(nr)
for i in range(len(preds)):
    if preds[i] > threshold:
        preds_norm[i] = 1
    else:
        preds_norm[i] = 0

sims_norm = [1 if float(x) > 0.5 else 0 for x in sims]
result = [str(x*y) for x, y in zip(preds_norm,sims_norm)]
print '\n'.join(result)

# data = pd.read_csv('New_data/csv/data_sub.csv',sep="#",header = 0, index_col=False)
# features = ['Mas','Ecs_1','Ecs_2','Ecs_3','Snp','Cnf_1','Cnf_2','Cnf_3','Cnf_4','Sub','Sen','Slc']
# label = ['Label_wp']
# preds_norm_all = []
# labels = []
# for i in range(1,6740):
#     data_sub = data.loc[data['index'] == i]
#     dtest = xgb.DMatrix(data_sub[features],data_sub[label])
#     nr = dtest.num_row()

#     preds = np.zeros(nr)
#     try:
#         for bst in bsts:
#             p = bst.predict_proba(data_sub[features])
#             l = np.array([ 1 if i[1] > i[0] else 0 for i in p])
#             preds = np.add(preds,l)

#         preds_norm = np.zeros(nr)
#         for i in range(len(preds)):
#                 if preds[i] > 14:
#                     preds_norm[i] = 1
#                 else:
#                     preds_norm[i] = 0
#         preds_norm_all.append(np.amax(preds_norm))
#         labels.append(dtest.get_label()[0])
#     except:
#         preds_norm_all.append(1)
#         labels.append(0)

# a = calculate_percison(preds_norm_all,labels)
# r = calculate_recall(preds_norm_all,labels)
# if a == 0 and r == 0:
#     f = 0
# else:
#     f = 2*(a*r)/(a+r)
# print 'precision: {}, recall: {}, F1 score: {}'.format(str(a),str(r),str(f))