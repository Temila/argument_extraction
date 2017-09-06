import xgboost as xgb
import requests, json
import pandas as pd
from nltk.tokenize import sent_tokenize
from util import read_essay_txt_file, read_essay_ann_file
# from sequential_pattern import sequential_pattern
from sentence_component import Sentence_component
from sematic_similarity import Sematic_Similarity

# sp = sequential_pattern()
# ss = Sematic_Similarity()
#load xgboost model
bst = xgb.Booster(model_file='New_data/claim_classifier_sequence.model')
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

data = pd.read_csv('New_data/csv/data_seq_np.csv',header = 0)
features = []
for i in range(134):
    features.append(str(i+1))
label = ['Label']
dtest = xgb.DMatrix(data[features],data[label])

print data[features]

preds = bst.predict(dtest)
labels = dtest.get_label()
preds_norm = [1 if x > 0.5 else 0 for x in preds]
a = calculate_percison(preds_norm,labels)
r = calculate_recall(preds_norm,labels)
if a == 0 and r == 0:
    f = 0
else:
    f = 2*(a*r)/(a+r)
print 'precision: {}, recall: {}, F1 score: {}'.format(str(a),str(r),str(f))