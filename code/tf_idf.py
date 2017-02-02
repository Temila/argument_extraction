#!/usr/bin/env python
# encoding: utf-8

"""
File: tfidf.py
Author: Harry Schwartz
Date: Dec 2010
The simplest TF-IDF library imaginable.
Add your documents as two-element lists 
[docname, [list_of_words_in_the_document] ] with 
addDocument (docname, list_of_words).  Get a list of all the 
[docname, similarity_score] pairs relative to a document by 
calling similarities ( [list_of_words] ).
See README.txt for a usage example.
"""
from nltk.tokenize import word_tokenize
import collections
import operator
import math

class TF_IDF:

    def __init__(self, Documents_1, Documents_2):
        self.Documents_1 = Documents_1
        self.Documents_2 = Documents_2
        self.tf_score = self._get_tf_score()
        self.idf_score = self._get_idf_score()

    def _get_tf_score(self):
        vocabulary = []
        for document in self.Documents_1:
            vocabulary.extend(word_tokenize(document))
        return self._calculate_tf_score(vocabulary)

    def _calculate_tf_score(self, listOfWords):
        Dict = collections.defaultdict(int)
        for w in listOfWords:
            Dict[w] += + 1.0
                
        # normalizing the query
        length = float (len (listOfWords))
        TFDict = {}
        for x in Dict:
            TFDict[x] = Dict[x] / length

        return TFDict

    def _get_idf_score(self):
        Dict = collections.defaultdict(int)
        allDoc = []
        allDoc.extend(self.Documents_1)
        allDoc.extend(self.Documents_2)
        length = len(allDoc)
        for word in self.tf_score:
            for doc in allDoc:
                if word in doc:
                    Dict[word] += 1.0
        IDFDict = {}
        for x in Dict:
            IDFDict[x] = math.log(length / (Dict[x] + 1))

        return IDFDict

    def get_tf_idf_score(self):
        tf_idf_dict = {}
        for x in self.tf_score:
            try:
                tf_idf_dict[x] = self.tf_score[x] * self.idf_score[x]
            except:
                continue
        return tf_idf_dict

# tfidf = TF_IDF(['this is sentence','this is also sentence','last sentence'],['sentence','sentence','doc'])
# print tfidf.get_tf_idf_score()