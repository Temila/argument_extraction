name = 'Amateur_boxing'
file = 'data/wiki12_articles/' + name
num = 8
from nltk.tokenize import sent_tokenize
from sentence_component import Sentence_component


with open(file,'r') as f:
    sentences = sent_tokenize(f.read())

sentence = sentences[num - 1]
sc = Sentence_component(sentence,'boxing',None)
print sentence
print len(sc.sentence_output['sentences'])
print sc._get_subjects()