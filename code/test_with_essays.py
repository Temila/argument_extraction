import xgboost as xgb
import requests, json
import pandas as pd
from nltk.tokenize import sent_tokenize
from util import read_essay_txt_file, read_essay_ann_file
# from sequential_pattern import sequential_pattern
from sentence_component import Sentence_component
from sematic_similarity import Sematic_Similarity
from sklearn.externals import joblib

# sp = sequential_pattern()
ss = Sematic_Similarity()
#load xgboost model
# bst = xgb.Booster(model_file='New_data/claim_classifier.model')
bst = joblib.load('my_model.pkl')
print 'model loaded'
accuracy = []
recall = []
f1 = []

def calculate_accuracy(detected_claims, claims):
    count = 0
    for detected_claim in detected_claims:
        for claim in claims:
            if claim in detected_claim:
                count += 1
                break
    try:
        return float(count) / len(detected_claims)
    except:
        return 0.0

def calculate_recall(detected_claims, claims):
    count = 0
    for claim in claims:
        for detected_claim in detected_claims:
            if claim in detected_claim:
                count += 1
                break
    return float(count) / len(claims)+0.000001

for i in range(1,403):
    topic, data = read_essay_txt_file(i)
    claims = read_essay_ann_file(i)
    sentences = sent_tokenize(data)
    #encode sentence
    headers = ['Mas','Ecs_1','Ecs_2','Ecs_3','Snp','Cnf_1','Cnf_2','Cnf_3','Cnf_4','Sub','Sen','Slc']
    all_setence_features = []
    print 'processing essay {}, {} sentences found'.format(str(i),len(sentences))
    index = 0
    for sentence in sentences:
        index += 1
        sc = Sentence_component(sentence,topic,ss)
        features = sc.extract_features()[1:]
        all_setence_features.append(features)
        print 'sentence {} done'.format(index)
    # dtest = xgb.DMatrix(all_setence_features)
    dtest = pd.DataFrame(all_setence_features, columns=headers)
    print dtest
    preds = bst.predict(dtest[headers])
    detected_claims = []
    for i in range(len(preds)):
        if preds[i] > 0.5:
            detected_claims.append(sentences[i])
    a = calculate_accuracy(detected_claims,claims)
    r = calculate_recall(detected_claims,claims)
    if a == 0 and r == 0:
        f = 0
    else:
        f = 2*(a*r)/(a+r)
    print 'accuracy: {}, recall: {}, F1 score: {}'.format(str(a),str(r),str(f))
    with open('result_4.csv','a+') as file:
        f.write('{},{},{},{}\n'.format(str(i),str(a),str(r),str(f)))
    # accuracy.append(str(a))
    # recall.append(str(r))
    # f1.append(str(f))

# str_accuracy = ','.join(accuracy)
# str_recall = ','.join(recall)
# str_f1 = ','.join(f1)

# with open('result_2.txt', 'w') as f:
#     f.write(str_accuracy+'\n'+str_recall+'\n'+str_f1)
