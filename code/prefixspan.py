import sys, json, operator
from collections import defaultdict

# db = [
#     [0, 1, 2, 3, 4],
#     [1, 1, 1, 3, 4],
#     [2, 1, 2, 2, 0],
#     [1, 1, 1, 2, 2],
# ]

# db = [[['Amateur', 'JJ', None, None, None], ['boxing', 'NN', None, None, 'topic'], ['is', 'VBZ', 'sentiment', 'claim', None], ['practised', 'VBN', None, None, None], ['at', 'IN', 'sentiment', None, None], ['the', 'DT', 'sentiment', 'claim', None], ['collegiate', 'JJ', None, None, None], ['level', 'NN', None, None, None], ['at', 'IN', 'sentiment', None, None], ['the', 'DT', 'sentiment', 'claim', None], ['Olympic', 'NNP', None, None, None], ['Games', 'NNPS', None, None, None], ['and', 'CC', 'sentiment', 'claim', None], ['Commonwealth', 'NNP', None, None, None], ['Games', 'NNPS', None, None, None], ['and', 'CC', 'sentiment', 'claim', None], ['in', 'IN', 'sentiment', 'claim', 'topic'], ['many', 'JJ', None, 'claim', None], ['other', 'JJ', 'sentiment', 'claim', None], ['venues', 'NN', None, None, None], ['sponsored', 'VBN', None, None, None], ['by', 'IN', 'sentiment', 'claim', None], ['amateur', 'JJ', None, None, None], ['boxing', 'NN', None, None, 'topic'], ['associations', 'NN', None, None, None]]]


class prefixspan:
    def __init__(self,db,threshold_ratio):
        self.db = db
        self.length = len(db)
        print '{} sentences found.'.format(self.length)
        self.threshold = threshold_ratio * self.length
        print 'threshold is set to {}'.format(self.threshold)
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
                            update = True
                            for j in range(len(this_tier_sequence_dict[word[i]])):
                                if sentence == this_tier_sequence_dict[word[i]][j]['sentence']:
                                    update = False
                                    break
                            if update:
                                this_tier_sequence_dict[word[i]].append({'sentence':sentence, 'index':index})

                            # this_tier_sequence_dict[word[i]] = [{'sentence':sentence, 'index':index}]
                        else:
                            this_tier_sequence_dict[word[i]] = [{'sentence':sentence, 'index':index}]
        else:
            sequences = last_tier_sequence_dict.keys()
            for sequence in sequences:
                sentences = last_tier_sequence_dict[sequence]
                for sentence_with_index in sentences:
                    sentence = sentence_with_index['sentence']
                    end_index = sentence_with_index['index'] + 1
                    for index, word in enumerate(sentence[end_index:]):
                        for i in range(1,5):
                            if word[i] is None:
                                continue
                            else:
                                new_sequence = sequence+','+word[i]
                                if new_sequence in this_tier_sequence_dict:
                                    update = True
                                    for j in range(len(this_tier_sequence_dict[new_sequence])):
                                        if sentence == this_tier_sequence_dict[new_sequence][j]['sentence']:
                                            # this_tier_sequence_dict[word[i]][j]['index'] = index+end_index
                                            update = False
                                            break
                                    if update:
                                        this_tier_sequence_dict[new_sequence].append({'sentence':sentence, 'index':index+end_index})
                                else:
                                    this_tier_sequence_dict[new_sequence] = [{'sentence':sentence,'index':index+end_index}]
        
        for sequence in this_tier_sequence_dict.keys():
            sentences = this_tier_sequence_dict[sequence]
            if len(sentences) < self.threshold:
                del this_tier_sequence_dict[sequence]

        self.results.append(this_tier_sequence_dict)

        if l < max_length:
            self.mining_sequence(this_tier_sequence_dict, l)

    def parse_results(self):
        # print 'parsing result'
        result = {}
        # for tiers in self.results[1:]:
        tiers = self.results[2]
        for sequence in tiers.keys():
            result[sequence] = float(len(tiers[sequence]))/self.length
        return sorted(result.items(), key=operator.itemgetter(1), reverse = True)   

    def parse_results_unsorted(self):
        # print 'parsing result'
        result = {}
        # for tiers in self.results[1:]:
        tiers = self.results[2]
        for sequence in tiers.keys():
            result[sequence] = float(len(tiers[sequence]))/self.length
        return result

# with open('New_data/encoded_none_claims.txt','r') as f:
#     db = json.load(f)[:1]
# pf = prefixspan(db,0.5)
# pf.mining_sequence({},0)
# result = pf.parse_results()
# print result
# with open('New_data/sequence_none_claim.txt','w') as f:
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

