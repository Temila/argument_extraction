import requests, json
from pycorenlp import StanfordCoreNLP
from nltk.tokenize import sent_tokenize
from util import read_essay_txt_file, read_essay_ann_file
import string

nlp = StanfordCoreNLP('http://localhost:9000')
printable = set(string.printable)

def constituency_parse_tree(sentence):
        return " ".join(nlp.annotate(sentence, properties={
            'annotators': 'parse',
            'outputFormat': 'json'
        })['sentences'][0]['parse'].replace('\n','').split())

def is_claim(sentence, claims):
    for claim in claims:
        if claim in sentence or claim == sentence:
            return 1
        else:
            pass
    return -1

def clean(sentence):
    return filter(lambda x: x in printable, sentence)

def form_data(tree, label):
        temp = '{} \t|BT| {} |ET|'.format(str(label),tree)
        return temp

output = []
for i in range(1,403):
    topic, data = read_essay_txt_file(i)
    claims = read_essay_ann_file(i)
    sentences = sent_tokenize(clean(data))
    for sentence in sentences:
        tree = constituency_parse_tree(sentence)
        label = is_claim(sentence, claims)
        line = form_data(tree, label)
        print line
        output.append(line)

with open('New_data/tree_essay_wp.test','w') as f:
    f.write('\n'.join(output))
