from pycorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('http://localhost:9000')
text = (
    'Pusheen and Smitha walked along the beach. Pusheen wanted to surf,'
    'but fell off the surfboard.')
output = nlp.annotate(text, properties={
    'annotators': 'tokenize,ssplit,pos,depparse,parse,openie',
    'outputFormat': 'json'
})
subjects = []
for x in output['sentences'][1]['openie']:
    subjects.append(x['subject'])
print list(set(subjects))