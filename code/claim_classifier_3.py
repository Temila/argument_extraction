import xgboost as xgb
import pandas as pd
import numpy as np
import operator
from util import generate_train_test, generate_train_test_2, balance_data

def calculate_accuracy(preds, labels):
    count = len(preds)
    a = 0.0
    b = 0.0
    for i in range(count):
        if preds[i] == 0:
            a += 1
            if labels[i] == 0:
                b += 1
    return float(b)/float(a)

def calculate_recall(preds, labels):
    count = len(preds)
    a = 0.0
    b = 0.0
    for i in range(count):
        if labels[i] == 0:
            a += 1
            if preds[i] == 0:
                b += 1
    return float(b)/float(a)

data_df_raw = pd.read_csv('New_data/csv/data_2.csv',header = 0)
data_df, data_test = generate_train_test(data_df_raw)
features = ['Mas','Ecs_1','Ecs_2','Ecs_3','Snp','Cnf_1','Cnf_2','Cnf_3','Cnf_4','Sub','Sen','Slc']
label = ['Label']
bsts = []
for i in range(1,28):
    train_df, test_df = generate_train_test_2(data_df,3000,i)
    dtrain = xgb.DMatrix(train_df[features],train_df[label])
    dtest = xgb.DMatrix(test_df[features],test_df[label])
    param = {'max_depth':15, 'eta':0.1, 'silent':1, 'objective':'binary:logistic','n_estimators':1000}
    # watchlist = [(dtest, 'eval'), (dtrain, 'train')]
    labels = dtrain.get_label()
    ratio = float(np.sum(labels==0)) / np.sum(labels==1)
    param['scale_pos_weight'] = ratio
    num_round = 1000
    # res = xgb.cv(param, dtrain, num_round, nfold=10,
    #        metrics={'auc'}, seed = 0,
    #        early_stopping_rounds=10,
    #        callbacks=[xgb.callback.print_evaluation(show_stdv=True)])
    num_round = 100
    bst = xgb.train(param, dtrain, num_round)
    # bst.save_model('New_data/unbalanced/claim_classifier_{}.model'.format(i))
    ptrain = bst.predict(dtrain, output_margin=True)
    ptest  = bst.predict(dtest, output_margin=True)
    dtrain.set_base_margin(ptrain)
    dtest.set_base_margin(ptest)
    data = train_df.append(test_df)
    dtrain = xgb.DMatrix(data[features], data[label])
    bst = xgb.train( param, dtrain, num_round)
    bsts.append(bst)

dtest = xgb.DMatrix(data_test[features],data_test[label])
nr = dtest.num_row()
preds = np.zeros(nr)
for bst in bsts:
    p = bst.predict(dtest)
    l = np.array([ 1 if i > 0.5 else 0 for i in p])
    preds = np.add(preds,l)

preds_norm = [1 if x > 14 else 0 for x in preds]
ls = dtest.get_label()

print calculate_accuracy(preds, ls)
print calculate_recall(preds, ls)


print ('error=%f' % ( sum(1 for i in range(len(preds)) if int(preds[i]>14)!=ls[i]) /float(len(preds))))

