from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import operator
from util import generate_train_test, generate_train_test_2, balance_data

def calculate_percision(preds, labels):
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
    if a == 0:
        return 0
    else:
        return float(b)/float(a)

def calculate_f1(percision, recall):
    return 2 * (percision * recall) / (percision + recall)

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

data_df_raw = pd.read_csv('New_data/csv/data_2.csv',header = 0)
features = ['Mas','Ecs_1','Ecs_2','Ecs_3','Snp','Cnf_1','Cnf_2','Cnf_3','Cnf_4','Sub','Sen','Slc']
label = ['Label']
# data_df_raw = pd.read_csv('New_data/csv/sequences_data_full.csv',header = 0)
# features = []
# for i in range(134):
#     features.append(str(i+1))
# label = ['Label']
f1 = []
train_df, test_df = generate_train_test(data_df_raw)
train_df = balance_data(train_df,features,label)
for n in range(3,61):
    neigh = KNeighborsClassifier(n_neighbors=n, algorithm='auto')
    neigh.fit(train_df[features],train_df[label]) 
    preds = neigh.predict_proba(test_df[features])
    labels = test_df[label].values.tolist()
    labels = [x[0] for x in labels]
    p = [1 if x[0] < x[1] else 0 for x in preds]
    percision = calculate_percision(p,labels)
    recall = calculate_recall(p, labels)
    f1.append(calculate_f1(percision, recall))
    print 'K={},p={},r={}'.format(str(n), str(percision),str(recall))
f1 = np.array(f1)
best_k = np.argmax(f1)+3
print 'best k: {}'.format(str(best_k))