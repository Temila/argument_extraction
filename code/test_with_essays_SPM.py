import xgboost as xgb
import requests, json
import pandas as pd
from nltk.tokenize import sent_tokenize
from util import read_essay_txt_file, read_essay_ann_file, match_sequences
from sequential_pattern import sequential_pattern
# from sentence_component import Sentence_component
# from sematic_similarity import Sematic_Similarity
# from sklearn.externals import joblib

sp = sequential_pattern()
# ss = Sematic_Similarity()
#load xgboost model
bst = xgb.Booster(model_file='New_data/claim_classifier_sequence.model')
# bst = joblib.load('my_model.pkl')
print 'model loaded'
accuracy = []
recall = []
f1 = []

with open('New_data/all_sequence.txt','r') as f:
    sequences = json.load(f)

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
    #encode 
    headers = []
    for i in range(134):
        headers.append(str(i+1))
    label = ['Label']
    
    all_setence_features = []
    print 'processing essay {}, {} sentences found'.format(str(i),len(sentences))
    index = 0
    for sentence in sentences:
        index += 1
        es = sp.encode_sentence(sentence, topic)
        features = match_sequences(es,sequences)
        # print features
        all_setence_features.append(features)
        print 'sentence {} done'.format(index)
    dtest = xgb.DMatrix(all_setence_features)
    # dtest = pd.DataFrame(all_setence_features, columns=headers)
    print dtest
    preds = bst.predict(dtest)
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
    with open('New_data/result/test_with_essay/result_SPM.csv','a+') as file:
        file.write('{},{},{},{}\n'.format(str(i),str(a),str(r),str(f)))
