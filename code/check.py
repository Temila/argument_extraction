import requests, json
import pandas as pd
from nltk.tokenize import sent_tokenize
from util import read_essay_txt_file, read_essay_ann_file


for i in range(1,403):
    topic, data = read_essay_txt_file(i)
    claims = read_essay_ann_file(i)
    sentences = sent_tokenize(data)
    print '***********************'
    print i
    print len(sentences)
    print len(claims)
