import xgboost as xgb
import requests, json
from nltk.tokenize import sent_tokenize
from util import read_essay_txt_file, read_essay_ann_file
# from sequential_pattern import sequential_pattern
from sentence_component import Sentence_component
from sematic_similarity import Sematic_Similarity

# sp = sequential_pattern()
ss = Sematic_Similarity()
#load xgboost model
bst = xgb.Booster(model_file='New_data/claim_classifier.model')

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
    return float(count) / len(detected_claims)

def calculate_recall(detected_claims, claims):
    count = 0
    for claim in claims:
        for detected_claim in detected_claims:
            if claim in detected_claim:
                count += 1
                break
    return float(count) / len(claims)

for i in range(1,403):
    topic, data = read_essay_txt_file(i)
    claims = read_essay_ann_file(i)
    sentences = sent_tokenize(data)
    #encode sentence
    all_setence_features = []
    print 'processing essay {}, {} sentences found'.format(str(i),len(sentences))
    index = 0
    for sentence in sentences:
        index += 1
        sc = Sentence_component(sentence,topic,ss)
        features = sc.extract_features()[1:]
        all_setence_features.append(features)
        print 'sentence {} done'.format(index)
    dtest = xgb.DMatrix(all_setence_features)
    preds = bst.predict(dtest)
    detected_claims = []
    for i in range(len(preds)):
        if preds[i] > 0.5:
            detected_claims.append(sentences[i])
    a = calculate_accuracy(detected_claims,claims)
    r = calculate_recall(detected_claims,claims)
    f = 2*(a*r)/(a+r)
    print 'accuracy: {}, recall: {}, F1 score: {}'.format(str(a),str(r),str(f))
    accuracy.append(str(a))
    recall.append(str(r))
    f1.append(str(f))

str_accuracy = ','.join(accuracy)
str_recall = ','.join(recall)
str_f1 = ','.join(f1)

with open('result.txt', 'w') as f:
    f.write(str_accuracy+'\n'+str_recall+'\n'+str_f1)
