import xgboost as xgb
import pandas as pd
import numpy as np
import operator
from util import leave_one_topic_out_train_test, balance_data

data_df_raw = pd.read_csv('New_data/csv/data_2.csv',header = 0)
features = ['Mas','Ecs_1','Ecs_2','Ecs_3','Snp','Cnf_1','Cnf_2','Cnf_3','Cnf_4','Sub','Sen','Slc']
label = ['Label']
topics = data_df_raw.Topic.unique()
count = 0
for topic in topics:
    count += 1
    print 'topic: {} is used as the test topic'.format(topic)
    # data_df_raw = pd.read_csv('New_data/csv/data_2.csv',header = 0)
    # train_df, test_df = leave_one_topic_out_train_test(data_df_raw,topic)
    # data_train = balance_data(train_df,features,label)
    # # data_test = balance_data(test_df,features,label)
    # dtrain = xgb.DMatrix(data_train[features],data_train[label])
    # dtest = xgb.DMatrix(test_df[features],test_df[label])
    # param = {'max_depth':10, 'eta':0.1, 'silent':1, 'objective':'binary:logistic'}
    # num_round = 100
    # l = dtrain.get_label()
    # ratio = float(np.sum(l == 0)) / np.sum(l==1)
    # param['scale_pos_weight'] = ratio
    # bst = xgb.train(param, dtrain, num_round)
    # # bst.save_model('New_data/claim_classifier.model')
    # ptrain = bst.predict(dtrain, output_margin=True)
    # ptest  = bst.predict(dtest, output_margin=True)
    # dtrain.set_base_margin(ptrain)
    # dtest.set_base_margin(ptest)
    # bst = xgb.train( param, dtrain, 1)
    # preds = bst.predict(dtest)
    # labels = dtest.get_label()
    # print 'calculate recall'
    # count = 0.0
    # count_1 = 0.0
    # for i in range(len(labels)):
    #     if labels[i] == 1:
    #         count_1 += 1
    #         if preds[i] > 0.5:
    #             count += 1
    # recall = float(count) / count_1

    # print 'calculate precision'
    # count = 0.0
    # count_1 = 0.0
    # for i in range(len(labels)):
    #     if preds[i] > 0.5:
    #         count_1 += 1
    #         if labels[i]  == 1:
    #             count += 1
    # percision = float(count) / count_1
    # temp_str = '{},{},{},{}\n'.format(topic,str(len(labels)),str(percision),str(recall))
    # with open('leave_one_topic_out_test_result_2.csv','a+') as f:
    #     f.write(temp_str)
print count