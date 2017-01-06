import nltk
from gensim import matutils
import math
import numpy as np
from glove import Glove

class Sematic_Similarity:

    def __init__(self):
        self.model = Glove.load_stanford('glove.6B.50d.txt')
        self.dictionary = self.model.dictionary
        self.word_vectors = self.model.word_vectors

######################### word similarity ##########################

    def cosine_similarity(self,v1,v2):
        "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(v1)):
            x = v1[i]; y = v2[i]
            sumxx += x*x
            sumyy += y*y
            sumxy += x*y
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

        return self.word_distance(v_1,v_2)

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