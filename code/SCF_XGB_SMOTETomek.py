import xgboost as xgb
import requests, json
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
from util import read_essay_txt_file, read_essay_ann_file
# from sequential_pattern import sequential_pattern
from sentence_component import Sentence_component
from sematic_similarity import Sematic_Similarity

# sp = sequential_pattern()
# ss = Sematic_Similarity()
#load xgboost model
bst = xgb.Booster(model_file='New_data/claim_classifier_2.model')
print 'model loaded'
accuracy = []
recall = []
f1 = []

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

def read_sim():
    with open('New_data/sims_essay','r') as f:
        return f.read().split('\n')

data = pd.read_csv('New_data/csv/data_wp.csv',sep="#",header = 0)
features = ['Mas','Ecs_1','Ecs_2','Ecs_3','Snp','Cnf_1','Cnf_2','Cnf_3','Cnf_4','Sub','Sen','Slc']
label = ['Label']
dtest = xgb.DMatrix(data[features],data[label])

preds = bst.predict(dtest)
sims = read_sim()
labels = dtest.get_label()
preds_norm = [1 if x > 0.5 else 0 for x in preds]
sims_norm = [1 if float(x) > 0.5 else 0 for x in sims]
result = [str(x*y) for x, y in zip(preds_norm,sims_norm)]
print '\n'.join(result)
a = calculate_percison(result,labels)
r = calculate_recall(result,labels)
if a == 0 and r == 0:
    f = 0
else:
    f = 2*(a*r)/(a+r)
# print 'precision: {}, recall: {}, F1 score: {}'.format(str(a),str(r),str(f))

# data = pd.read_csv('New_data/csv/data_sub_2.csv',header = 0, index_col=False)
# features = ['Mas','Ecs_1','Ecs_2','Ecs_3','Snp','Cnf_1','Cnf_2','Cnf_3','Cnf_4','Sub','Sen','Slc']
# label = ['Label_wp']
# preds_norm = []
# labels = []
# for i in range(1,6740):
#     data_sub = data.loc[data['index'] == i]
#     # print data_sub
#     dtest = xgb.DMatrix(data_sub[features],data_sub[label])
#     try:
#         preds_norm.append(round(np.amax(bst.predict(dtest))))
#         labels.append(dtest.get_label()[0])
#     except:
#         preds_norm.append(1)
#         labels.append(0)

# for i in preds_norm:
#     print i

# a = calculate_percison(preds_norm,labels)
# r = calculate_recall(preds_norm,labels)
# if a == 0 and r == 0:
#     f = 0
# else:
#     f = 2*(a*r)/(a+r)
# print 'precision: {}, recall: {}, F1 score: {}'.format(str(a),str(r),str(f))