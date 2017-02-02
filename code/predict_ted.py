import xgboost as xgb
import requests, json
from nltk.tokenize import sent_tokenize
from util import match_sequences
# from sequential_pattern import sequential_pattern
from sentence_component import Sentence_component
from sematic_similarity import Sematic_Similarity

# sp = sequential_pattern()
ss = Sematic_Similarity()
#load xgboost model
bst = xgb.Booster(model_file='New_data/claim_classifier.model')
#set topics
# query = ['video games']
# #find relevant ted talk
# url = 'https://document-elasticsearch.herokuapp.com/search/ted'
# params = {
#     'index':'ted',
#     'topics':query,
#     'field':'transcript'
#     }
# recommendation_list = requests.post(url,json=params).json()
#read file and tokenize sentences
# files = [x[0].replace(' ','_') for x in recommendation_list]
with open('essay001.txt','r') as f:
    data = f.read()
sentences = sent_tokenize(data)
#target sequences
# with open('New_data/all_sequence.txt','r') as f:
#     sequences = json.load(f)
#encode sentence
all_setence_features = []
print 'processing {}, {} sentences found'.format('essay001',len(sentences))
index = 0
for sentence in sentences:
    index += 1
    # encoded_sentence = sp.encode_sentence(sentence,query)
    # seq_features = match_sequences(encoded_sentence,sequences)
    sc = Sentence_component(sentence,'Should students be taught to compete or to cooperate',ss)
    features = sc.extract_features()[1:]
    all_setence_features.append(features)
    print 'sentence {} done'.format(index)
dtest = xgb.DMatrix(all_setence_features)
preds = bst.predict(dtest)
print preds
for i in range(len(preds)):
    if preds[i] > 0.5:
        print sentences[i]
# print ('error=%f' % ( sum(1 for i in range(len(preds)) if int(preds[i]>0.5)!=labels[i]) /float(len(preds))))