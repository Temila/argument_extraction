import xgboost as xgb
import pandas as pd
import numpy as np
import operator
from util import generate_train_test, balance_data

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

data_df = pd.read_csv('New_data/csv/data_2.csv',header = 0)
features = ['Mas','Ecs_1','Ecs_2','Ecs_3','Snp','Cnf_1','Cnf_2','Cnf_3','Cnf_4','Sub','Sen']
label = ['Label']
# print data_df
train_df, test_df = generate_train_test(data_df)
# train_df = balance_data(train_df,features,label)
# dtrain = xgb.DMatrix(data_df[features],data_df[label])
dtrain = xgb.DMatrix(train_df[features],train_df[label])
dtest = xgb.DMatrix(test_df[features],test_df[label])
param = {'max_depth':10, 'eta':0.1, 'silent':1, 'objective':'binary:logistic','n_estimators':100}
# watchlist = [(dtest, 'eval'), (dtrain, 'train')]
label = dtrain.get_label()
ratio = float(np.sum(label == 0)) / np.sum(label==1)
param['scale_pos_weight'] = ratio
num_round = 1000
# res = xgb.cv(param, dtrain, num_round, nfold=10,
#        metrics={'auc'}, seed = 0,
#        early_stopping_rounds=10,
#        callbacks=[xgb.callback.print_evaluation(show_stdv=True)])
print 'training'
bst = xgb.train(param, dtrain, num_round)
bst.save_model('New_data/claim_classifier_feature_unbalance.model')
# ptrain = bst.predict(dtrain, output_margin=True)
# ptest  = bst.predict(dtest, output_margin=True)
# dtrain.set_base_margin(ptrain)
# dtest.set_base_margin(ptest)
# bst = xgb.train( param, dtrain, 1)
preds = bst.predict(dtest)
labels = dtest.get_label()
preds_norm = [1 if x > 0.5 else 0 for x in preds]

print calculate_percison(preds_norm, labels)
print calculate_recall(preds_norm, labels)

# print ('error=%f' % ( sum(1 for i in range(len(preds)) if int(preds[i]>0.5)!=labels[i]) /float(len(preds))))