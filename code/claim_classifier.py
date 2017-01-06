import xgboost as xgb
import pandas as pd
import numpy as np
from util import generate_train_test

# data_df = pd.read_csv('data.csv',header = 0)
# true_label = data_df.loc[data_df['Label'] == 1]
# for i in range(35):
#     data_df = data_df.append(true_label)
# test_df = pd.read_csv('test.csv',header = 0)
# true_label = test_df.loc[test_df['Label'] == 1]
# for i in range(35):
#     test_df = test_df.append(true_label)
data_df = pd.read_csv('data/sentence_component.csv',header = 0)
train_df, test_df = generate_train_test(data_df)
features = ['Mas','Ecs_1','Ecs_2','Ecs_3','Cnf_1','Cnf_2','Cnf_3','Cnf_4','Sub','Sen','sq1','sq2','sq3','sq4','sq5','sq6','sq7']
label = ['Label']
dtrain = xgb.DMatrix(train_df[features],train_df[label])
dtest = xgb.DMatrix(test_df[features],test_df[label])
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic'}
label = dtrain.get_label()
ratio = float(np.sum(label == 0)) / np.sum(label==1)
param['scale_pos_weight'] = ratio
num_round = 100
res = xgb.cv(param, dtrain, num_round, nfold=10,
       metrics={'error'}, seed = 0,
       callbacks=[xgb.callback.print_evaluation(show_stdv=True)])
print res
bst = xgb.train(param, dtrain, num_round)
ptrain = bst.predict(dtrain, output_margin=True)
ptest  = bst.predict(dtest, output_margin=True)
dtrain.set_base_margin(ptrain)
dtest.set_base_margin(ptest)
bst = xgb.train( param, dtrain, 1)
preds = bst.predict(dtest)
labels = dtest.get_label()
print ('error=%f' % ( sum(1 for i in range(len(preds)) if int(preds[i]>0.5)!=labels[i]) /float(len(preds))))