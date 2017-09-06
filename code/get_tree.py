from pycorenlp import StanfordCoreNLP
from nltk.tokenize import sent_tokenize
from util import read_file
from util import read_essay_txt_file
from util import read_essay_ann_file
from util import read_essay_ann_file_no_premise
import numpy as np
import sys

class GetTree:

    def __init__(self):
        reload(sys)
        sys.setdefaultencoding('utf8')
        self.nlp = StanfordCoreNLP('http://localhost:9000')
        self.output_file = open('New_data/essays_tree_np','a')
        self.data = []

    def constituency_parse_tree(self, sentence):
        return " ".join(self.nlp.annotate(sentence, properties={
            'annotators': 'parse',
            'outputFormat': 'json'
        })['sentences'][0]['parse'].replace('\n','').split())

    def form_data(self, tree, label):
        temp = '{} \t|BT| {} |ET|'.format(str(label),tree)
        return temp

    def save_data(self,file):
        file.write('\n'.join(self.data))

    def generate_train_test(self):
        with open('New_data/tree_claim.txt','r') as f:
            data_true = f.read().split('\n')
        with open('New_data/tree_non_claim_all.txt','r') as f:
            data_false = f.read().split('\n')
        train_data_true, test_data_true = self._split(data_true,0.8)
        train_data_false, test_data_false = self._split(data_false,0.8)
        train_data = []
        test_data = []
        train_data.extend(train_data_true)
        train_data.extend(train_data_false)
        test_data.extend(test_data_true)
        test_data.extend(test_data_false)
        return train_data,test_data

    def generate_train(self):
        with open('New_data/tree_claim.txt','r') as f:
            data_true = f.read().split('\n')
        with open('New_data/tree_non_claim.txt','r') as f:
            data_false = f.read().split('\n')
        data = []
        data.extend(data_true)
        data.extend(data_false)
        return data


    def _split(self, data, ratio):
        data = np.array(data)
        msk = np.random.rand(len(data)) < ratio
        train_data = data[msk]
        test_data = data[~msk]
        return train_data, 

    def get_label(self, sentence, claims):
        for claim in claims:
            if claim in sentence:
                return "1"
        return "-1"

    def write_to_file(self, data):
        self.output_file.write(data)

gt = GetTree()

# non_claims_raw = read_file('New_data/non_claims.txt', cut=False)
# non_claims_cut = non_claims_raw[:10000]
# for x in non_claims_raw:
#     try:
#         sentence = x[1]
#         print sentence
#         cpt = gt.constituency_parse_tree(sentence)
#         gt.form_data(cpt, '0')
#     except:
#         pass
# with open('New_data/tree_non_claim_all.txt','w') as f:
#     gt.save_data(f)

# data = gt.generate_train()
# with open('New_data/tree_all.train', 'w') as f:
#     f.write('\n'.join(data))

for i in range(1,403):
    topic, data = read_essay_txt_file(i)
    claims = read_essay_ann_file_no_premise(i)
    sentences = sent_tokenize(data)
    for sentence in sentences:
        cpt = gt.constituency_parse_tree(sentence)
        l = gt.get_label(sentence, claims)
        output = gt.form_data(cpt,l)
        gt.write_to_file(output + '\n')
        print sentence


# with open('New_data/tree.train','w') as f:
#     f.write('\n'.join(train_data))
# with open('New_data/tree.test','w') as f:
#     f.write('\n'.join(test_data))
