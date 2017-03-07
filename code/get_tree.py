from pycorenlp import StanfordCoreNLP
from util import read_file
import numpy as np
import sys

class GetTree:

    def __init__(self):
        reload(sys)
        sys.setdefaultencoding('utf8')
        self.nlp = StanfordCoreNLP('http://localhost:9000')
        self.data = []

    def constituency_parse_tree(self, sentence):
        return " ".join(self.nlp.annotate(sentence, properties={
            'annotators': 'parse',
            'outputFormat': 'json'
        })['sentences'][0]['parse'].replace('\n','').split())

    def form_data(self, tree, label):
        temp = '{} \t|BT| {} |ET|'.format(str(label),tree)
        self.data.append(temp)

    def save_data(self,file):
        file.write('\n'.join(self.data))

    # def get_subset_tree(self,sentence_tree):
    #     pass

    # def get_sub_tree(self,sentence_tree):
    #     pass

    def generate_train_test(self):
        with open('New_data/tree_claim.txt','r') as f:
            data_true = f.read().split('\n')
        with open('New_data/tree_non_claim.txt','r') as f:
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


    def _split(self, data, ratio):
        data = np.array(data)
        msk = np.random.rand(len(data)) < ratio
        train_data = data[msk]
        test_data = data[~msk]
        return train_data, test_data

gt = GetTree()
train_data,test_data = gt.generate_train_test()
with open('New_data/tree_train.txt','w') as f:
    f.write('\n'.join(train_data))
with open('New_data/tree_test.txt','w') as f:
    f.write('\n'.join(test_data))
# claims = read_file('New_data/claims.txt')
# for claim in claims[:3000]:
#     try:
#         tree = gt.constituency_parse_tree(claim[1])
#     except:
#         continue
#     gt.form_data(tree,1)
# with open('New_data/tree_claim.txt','w') as f:
#     gt.save_data(f)

