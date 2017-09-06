from nltk.tokenize import sent_tokenize
from util import read_essay_txt_file, read_essay_ann_file

i = 3

def get_label(sentence, claims):
    for claim in claims:
        if claim in sentence:
            return 1
    return 0

output = ''
topic, data = read_essay_txt_file(i)
claims = read_essay_ann_file(i)
sentences = sent_tokenize(data)
for sentence in sentences:
    if get_label(sentence, claims) == 1:
        output += sentence + '#'
print output