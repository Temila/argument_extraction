import xgboost as xgb
import requests, json
from nltk.tokenize import sent_tokenize
from util import match_sequences
import operator
import re
# from sequential_pattern import sequential_pattern
from sentence_component import Sentence_component
from sematic_similarity import Sematic_Similarity

punctuation = '"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~'

def generate_sub_sentence(sentence):
    min_size = 4
    r = re.compile(r'[{}]'.format(re.escape(punctuation)))
    chunck_by_punc = r.split(sentence)
    candidates = [x for x in chunck_by_punc if len(x.split(' ')) >= min_size]
    return candidates

ss = Sematic_Similarity()
#load xgboost model

bst = xgb.Booster(model_file='New_data/claim_classifier_2.model')
topic = 'can we build a AI without losing control over it'
with open('essay001.txt','r') as f:
    data = f.read()
sentences = sent_tokenize(data)
all_setence_features = []
print 'processing {}, {} sentences found'.format('essay001',len(sentences))
index = 0
count = 0
match_list = {}
for sentence in sentences:
    index += 1
    sub_sentences = generate_sub_sentence(sentence)
    for sub_sentence in sub_sentences:
        count += 1
        sc = Sentence_component(sentence,topic,ss)
        features = sc.extract_features()[1:]
        all_setence_features.append(features)
        match_list[str(count)] = index
    print 'sentence {} done'.format(index)
dtest = xgb.DMatrix(all_setence_features)
preds = bst.predict(dtest)
detected_claims_raw = {}
for i in range(len(preds)):
    if preds[i] > 0.5:
        sen = sentences[int(match_list[str(i+1)])-1]
        if sen in detected_claims_raw.keys():
            last = detected_claims_raw[sen]
            detected_claims_raw[sen] = max(last,preds[i])
        else:
            detected_claims_raw[sen] = preds[i]

dc = sorted(detected_claims_raw.items(), key=operator.itemgetter(1), reverse = True)
print '\n\nrank with score'
dts = []
for d in dc:
    print d
    dts.append(d[0])
r_d = {}
for dt in dts:
    r_d[dt] = ss.n_similarity_2(dt,topic)
dc_2 = sorted(r_d.items(), key=operator.itemgetter(1), reverse = True)
print '\n\nrank with relatedness'
for d in dc_2:
    print d
