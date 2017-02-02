import xgboost as xgb
import pandas as pd
import numpy as np
import operator
from util import generate_train_test, balance_data

data_df_raw = pd.read_csv('New_data/csv/sequences_data_full.csv',header = 0)
features = []
for i in range(134):
    features.append(str(i+1))
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
# res = xgb.cv(param, dtrain, num_round, nfold=10,
#        metrics={'error'}, seed = 0,
#        callbacks=[xgb.callback.print_evaluation(show_stdv=True)])
# print res
bst = xgb.train(param, dtrain, num_round)
ptrain = bst.predict(dtrain, output_margin=True)
ptest  = bst.predict(dtest, output_margin=True)
dtrain.set_base_margin(ptrain)
dtest.set_base_margin(ptest)
bst = xgb.train( param, dtrain, 1)
preds = bst.predict(dtest)
labels = dtest.get_label()
print ('error=%f' % ( sum(1 for i in range(len(preds)) if int(preds[i]>0.5)!=labels[i]) /float(len(preds))))
f = open('xgb.fmap','w') 
for i in range(len(features)):
    f.write('{0}\t{1}\tq\n'.format(i, features[i]))
f.close()
importance = bst.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))
print importance

print 'number of test samples: {}'.format(len(labels), reverse=True)

print 'calculate recall'
count = 0.0
count_1 = 0.0
for i in range(len(labels)):
    if labels[i] == 1:
        count_1 += 1
        if preds[i] > 0.5:
            count += 1
print '{} / {}'.format(count, count_1)
print float(count) / count_1

print 'calculate precision'
count = 0.0
count_1 = 0.0
for i in range(len(labels)):
    if preds[i] > 0.5:
        count_1 += 1
        if labels[i]  == 1:
            count += 1
print '{} / {}'.format(count, count_1)
print float(count) / count_1