import nltk
from gensim import matutils
import math
import numpy as np
from glove import Glove
from nltk.corpus import stopwords

class Sematic_Similarity:

    def __init__(self):
        self.model = Glove.load_stanford('glove.6B.50d.txt')
        self.dictionary = self.model.dictionary
        self.word_vectors = self.model.word_vectors

######################### word similarity ##########################

    def cosine_similarity(self,v1,v2):
        "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
        sumxx = 0.0
        sumxy = 0.0
        sumyy = 0.0
        for i in range(len(v1)):
            x = v1[i]
            y = v2[i]
            sumxx += float(x*x)
            sumyy += float(y*y)
            sumxy += float(x*y)
        return sumxy/math.sqrt(sumxx*sumyy)

    def word_distance(self,v1,v2):
        sumxy = 0
        for i in range(len(v1)):
            x = v1[i]; y = v2[i]
            sumxy += (x-y)*(x-y)
        return math.sqrt(sumxy)


    def word_similarity(self, word_1, word_2):
        v_1 = self.word_vectors[self.dictionary[word_1]]
        v_2 = self.word_vectors[self.dictionary[word_2]]

        return self.cosine_similarity(v_1,v_2)

######################### overall similarity ##########################

    def n_similarity(self, ws1, ws2):
        v1 = []
        v2 = []
        for word in ws1:
            if "'s" in word:
                word = word.replace("'s",'')
            try:
                v1.append(self.word_vectors[self.dictionary[word.lower()]])
            except:
                continue
        for word in ws2:
            if "'s" in word:
                word = word.replace("'s",'')
            try:
                v2.append(self.word_vectors[self.dictionary[word.lower()]])
            except:
                continue
        if len(v1) == 0 or len(v2) == 0:
            return 0.0
        else:
            return np.dot(matutils.unitvec(np.array(v1).mean(axis=0)), matutils.unitvec(np.array(v2).mean(axis=0)))

    def one_way_similarity(self, s1, s2):
        stop = set(stopwords.words('english'))
        punctuation = '"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~'
        ws1 = [i for i in s1.lower().split(' ') if i not in stop and i not in punctuation]
        ws2 = [i for i in s2.lower().split(' ') if i not in stop and i not in punctuation]
        all_similarities = []
        for word_1 in ws1:
            if "'s" in word_1:
                word_1 = word_1.replace("'s","")
            similarities = []
            for word_2 in ws2:
                if "'s" in word_2:
                    word_2 = word_2.replace("'s","")
                # similarities.append(self.word_similarity(word_1,word_2))
                try:
                    similarities.append(self.word_similarity(word_1,word_2))
                except:
                    similarities.append(0.0)
            all_similarities.append(max(similarities))
        return float(sum(all_similarities)) / len(all_similarities)

    def n_similarity_2(self, s1, s2):
        return (self.one_way_similarity(s1,s2) + self.one_way_similarity(s2,s1))/2
