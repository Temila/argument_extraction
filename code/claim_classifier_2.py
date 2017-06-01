import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.externals import joblib
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from util import generate_train_test, balance_data


# import matplotlib.pylab as plt
# from matplotlib.pylab import rcParams
# rcParams['figure.figsize'] = 12, 4

data_df_raw = pd.read_csv('New_data/csv/data_2.csv',header = 0)
features = ['Mas','Ecs_1','Ecs_2','Ecs_3','Snp','Cnf_1','Cnf_2','Cnf_3','Cnf_4','Sub','Sen','Slc']
label = ['Label']
data_df = balance_data(data_df_raw,features,label)
# print data_df
# train_df, test_df = generate_train_test(data_df)
# dtrain = xgb.DMatrix(data_df[features],data_df[label])
# dtest = xgb.DMatrix(test_df[features],test_df[label])

def modelfit(alg, dtrain, features, label ,useTrainCV=True, cv_folds=10, early_stopping_rounds=50, save_model=False):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgb_param['silent'] = 1
        xgtrain = xgb.DMatrix(dtrain[features].values, label=dtrain[label].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics=list(['auc','error']), early_stopping_rounds=early_stopping_rounds, callbacks=[xgb.callback.print_evaluation(show_stdv=True)])
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[features], dtrain[label],eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[features])
    dtrain_predprob = alg.predict_proba(dtrain[features])[:,1]
        
    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(dtrain[label].values, dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[label], dtrain_predprob)
                    
    # feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    # feat_imp.plot(kind='bar', title='Feature Importances')
    # plt.ylabel('Feature Importance Score')

    if save_model:
        joblib.dump(alg, 'my_model.pkl', compress=9)

param_test = {
 'gamma':np.linspace(0.01,0.3,10)
}
print 'searching'
gsearch = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=1000, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=8, scale_pos_weight=1,seed=0,max_depth=9,min_child_weight=9), 
 param_grid = param_test, scoring='roc_auc',n_jobs=8,iid=False, cv=5, verbose=10)

gsearch.fit(data_df[features],data_df[label[0]])

# xgb1 = XGBClassifier(
#     learning_rate =0.1,
#     n_estimators=1000,
#     max_depth=9,
#     min_child_weight=9,
#     gamma=0,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     objective= 'binary:logistic',
#     nthread=4,
#     scale_pos_weight=1
#     )


# modelfit(xgb1, data_df, features, label, save_model=True)