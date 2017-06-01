from nltk import Tree
from pycorenlp import StanfordCoreNLP

def _parse_sentence(sentence):
    nlp = StanfordCoreNLP('http://localhost:9000')
    return nlp.annotate(sentence, properties={
        'annotators': 'tokenize,ssplit,pos,depparse,parse,ner',
        'outputFormat': 'json'
    })

def sub_tree_to_str(sentence):
    pt = _parse_sentence(sentence)['sentences'][0]['parse'][6:-1]
    tree = Tree.fromstring(pt)
    subtree_set = []
    for x in tree.subtrees():
        st = x.leaves()
        if len(st) > 3:
            subtree_set.append(' '.join(st))
    return subtree_set
        


sentence = 'However the Catalyst Model specifically states that media influences are too weak and distant to have much influence'
print '\n'.join(sub_tree_to_str(sentence))