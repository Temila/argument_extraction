import xgboost as xgb
import glob
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize

#load xgboost model
files = glob.glob('New_data/unbalanced/*.model')
bsts = []
for file in files:
    bsts.append(xgb.Booster(model_file=file))
print 'model loaded'

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

# data = pd.read_csv('New_data/csv/data_np.csv',sep="#",header = 0)
# features = ['Mas','Ecs_1','Ecs_2','Ecs_3','Snp','Cnf_1','Cnf_2','Cnf_3','Cnf_4','Sub','Sen','Slc']
# label = ['Label']
# dtest = xgb.DMatrix(data[features],data[label])
# labels = dtest.get_label()

# nr = dtest.num_row()
# preds = np.zeros(nr)
# sims = read_sim()
# for bst in bsts:
#     p = bst.predict(dtest)
#     l = np.array([ 1 if i > 0.5 else 0 for i in p])
#     preds = np.add(preds,l)

# preds_norm = np.zeros(nr)
# for i in range(len(preds)):
#     if preds[i] > 14:
#         preds_norm[i] = 1
#     else:
#         preds_norm[i] = 0
# sims_norm = [1 if float(x) > 0.5 else 0 for x in sims]
# result = [str(x*y) for x, y in zip(preds_norm,sims_norm)]
# print '\n'.join(result)

data = pd.read_csv('New_data/csv/data_sub_2.csv',header = 0, index_col=False)
features = ['Mas','Ecs_1','Ecs_2','Ecs_3','Snp','Cnf_1','Cnf_2','Cnf_3','Cnf_4','Sub','Sen','Slc']
label = ['Label_wp']
preds_norm_all = []
labels = []
for i in range(1,6740):
    data_sub = data.loc[data['index'] == i]
    dtest = xgb.DMatrix(data_sub[features],data_sub[label])
    nr = dtest.num_row()
    preds = np.zeros(nr)
    for bst in bsts:
        p = bst.predict(dtest)
        l = np.array([ 1 if x > 0.5 else 0 for x in p])
        preds = np.add(preds,l)
    preds_norm = np.zeros(nr)
    for j in range(len(preds)):
        if preds[j] > 14:
            preds_norm[j] = 1
        else:
            preds_norm[j] = 0
    try:
        preds_norm_all.append(np.amax(preds_norm))
        labels.append(dtest.get_label()[0])
    except:
        pass

for i in preds_norm_all:
    print i

a = calculate_percison(preds_norm_all,labels)
r = calculate_recall(preds_norm_all,labels)
if a == 0 and r == 0:
    f = 0
else:
    f = 2*(a*r)/(a+r)
print 'precision: {}, recall: {}, F1 score: {}'.format(str(a),str(r),str(f))