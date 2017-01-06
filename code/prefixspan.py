import sys, json, operator
from collections import defaultdict

# db = [
#     [0, 1, 2, 3, 4],
#     [1, 1, 1, 3, 4],
#     [2, 1, 2, 2, 0],
#     [1, 1, 1, 2, 2],
# ]

# db = [[['Amateur', 'JJ', None, None, None], ['boxing', 'NN', None, None, 'topic'], ['is', 'VBZ', 'sentiment', 'claim', None], ['practised', 'VBN', None, None, None], ['at', 'IN', 'sentiment', None, None], ['the', 'DT', 'sentiment', 'claim', None], ['collegiate', 'JJ', None, None, None], ['level', 'NN', None, None, None], ['at', 'IN', 'sentiment', None, None], ['the', 'DT', 'sentiment', 'claim', None], ['Olympic', 'NNP', None, None, None], ['Games', 'NNPS', None, None, None], ['and', 'CC', 'sentiment', 'claim', None], ['Commonwealth', 'NNP', None, None, None], ['Games', 'NNPS', None, None, None], ['and', 'CC', 'sentiment', 'claim', None], ['in', 'IN', 'sentiment', 'claim', 'topic'], ['many', 'JJ', None, 'claim', None], ['other', 'JJ', 'sentiment', 'claim', None], ['venues', 'NN', None, None, None], ['sponsored', 'VBN', None, None, None], ['by', 'IN', 'sentiment', 'claim', None], ['amateur', 'JJ', None, None, None], ['boxing', 'NN', None, None, 'topic'], ['associations', 'NN', None, None, None]]]

# with open('data/encoded_sentence_2.txt','r') as f:
#     db = json.load(f)
class prefixspan:
    def __init__(self,db):
        self.db = db
        self.length = len(db)
        print self.length
        self.threshold = 0
        self.results = []

    def mining_sequence(self, last_tier_sequence_dict, length, max_length = 3):
        l = length + 1
        print 'processing tier ' + str(l)
        this_tier_sequence_dict = {}
        if last_tier_sequence_dict == {} and length == 0:
            for sentence in self.db:
                for index, word in enumerate(sentence):
                    for i in range(1,5):
                        if word[i] is None:
                            continue
                        elif word[i] in this_tier_sequence_dict:
                            this_tier_sequence_dict[word[i]].append([sentence,index])
                        else:
                            this_tier_sequence_dict[word[i]] = [[sentence,index]]
        else:
            sequences = last_tier_sequence_dict.keys()
            for sequence in sequences:
                sentences = last_tier_sequence_dict[sequence]
                for sentence_with_index in sentences:
                    sentence = sentence_with_index[0]
                    end_index = sentence_with_index[1] + 1
                    for index, word in enumerate(sentence[end_index:end_index+3]):
                        for i in range(1,5):
                            if word[i] is None:
                                continue
                            else:
                                new_sequence = sequence+','+word[i]
                                if new_sequence in this_tier_sequence_dict:
                                    this_tier_sequence_dict[new_sequence].append([sentence,index+end_index])
                                else:
                                    this_tier_sequence_dict[new_sequence] = [[sentence,index+end_index]]
        
        for sequence in this_tier_sequence_dict.keys():
            sentences = this_tier_sequence_dict[sequence]
            if len(sentences) < self.threshold:
                del this_tier_sequence_dict[sequence]

        self.results.append(this_tier_sequence_dict)

        if l < max_length:
            self.mining_sequence(this_tier_sequence_dict, l)

    def parse_results(self):
        print 'parsing result'
        result = {}
        for tiers in self.results[1:]:
            for sequence in tiers.keys():
                result[sequence] = float(len(tiers[sequence]))/self.length
        return sorted(result.items(), key=operator.itemgetter(1), reverse = True)   

    def parse_results_unsorted(self):
        print 'parsing result'
        result = {}
        for tiers in self.results[1:]:
            for sequence in tiers.keys():
                result[sequence] = float(len(tiers[sequence]))/self.length
        return result


# pf = prefixspan(db)
# pf.mining_sequence({},0)
# result = pf.parse_results()
# print result
# with open('data/sequence_claim_2.txt','w') as f:
#     json.dump(result,f)

'''
{
    'sequence_1':   [
                      [
                        [words_1],index_1
                      ],
                      [
                        [words_2],index_2
                      ],
                      .
                      .
                      .
                      [
                        [words_n],index_n
                      ]
                    ],
    'sequence_2':   ...

}
'''

