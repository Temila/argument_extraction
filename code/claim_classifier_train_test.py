import xgboost as xgb
import pandas as pd
import numpy as np
from util import generate_train_test, balance_data

# data_df = pd.read_csv('data.csv',header = 0)
# true_label = data_df.loc[data_df['Label'] == 1]
# for i in range(35):
#     data_df = data_df.append(true_label)
# test_df = pd.read_csv('test.csv',header = 0)
# true_label = test_df.loc[test_df['Label'] == 1]
# for i in range(35):
#     test_df = test_df.append(true_label)

# def logregobj(preds, dtrain):
#     labels = dtrain.get_label()
#     preds = 1.0 / (1.0 + np.exp(-preds))
#     grad = preds - labels
#     hess = preds * (1.0-preds)
#     return grad, hess

# def evalerror(preds, dtrain):
#     labels = dtrain.get_label()
#     count_true = 0.0
#     count_false = 0.0
#     for i in range(len(labels)):
#         if preds[i] > 0.5:
#             if labels[i]  == 1:
#                 count_true += 1 * 0.7
#         else:
#             count_false += 1 * 0.3
#     return'error', float(count_true + count_false) / len(labels)


data_df_raw = pd.read_csv('New_data/csv/data_2.csv',header = 0)
features = ['Mas','Ecs_1','Ecs_2','Ecs_3','Snp','Cnf_1','Cnf_2','Cnf_3','Cnf_4','Sub','Sen','Slc']
label = ['Label']
data_df = balance_data(data_df_raw,features,label)
train_df, test_df = generate_train_test(data_df)
dtrain = xgb.DMatrix(train_df[features],train_df[label])
dtest = xgb.DMatrix(test_df[features],test_df[label])
param = {'max_depth':10, 'eta':0.1, 'silent':1, 'objective':'binary:logistic'}
watchlist = [(dtest, 'eval'), (dtrain, 'train')]
label = dtrain.get_label()
ratio = float(np.sum(label == 0)) / np.sum(label==1)
param['scale_pos_weight'] = ratio
num_round = 100

bst = xgb.train(param, dtrain, num_round)
ptrain = bst.predict(dtrain, output_margin=True)
ptest  = bst.predict(dtest, output_margin=True)
dtrain.set_base_margin(ptrain)
dtest.set_base_margin(ptest)
bst = xgb.train( param, dtrain, 1)
preds = bst.predict(dtest)
labels = dtest.get_label()
print ('error=%f' % ( sum(1 for i in range(len(preds)) if int(preds[i]>0.5)!=labels[i]) /float(len(preds))))

print 'calculate recall'
count = 0.0
count_1 = 0.0
for i in range(len(labels)):
    if labels[i] == 1:
        count_1 += 1
        if preds[i] > 0.5:
            count += 1
print count_1
print float(count) / count_1

print 'calculate precision'
count = 0.0
count_1 = 0.0
for i in range(len(labels)):
    if preds[i] > 0.5:
        count_1 += 1
        if labels[i]  == 1:
            count += 1
print count_1
print float(count) / count_1