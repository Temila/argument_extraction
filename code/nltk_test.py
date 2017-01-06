# from nltk.corpus import subjectivity
# from nltk.tokenize import regexp
# from nltk.sentiment import SentimentAnalyzer
# from nltk import NaiveBayesClassifier
# from copy import deepcopy
# import re, sys, random, pickle, codecs

# NEGATION = r"""
#     (?:
#         ^(?:never|no|nothing|nowhere|noone|none|not|
#             havent|hasnt|hadnt|cant|couldnt|shouldnt|
#             wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint
#         )$
#     )
#     |
#     n't"""

# NEGATION_RE = re.compile(NEGATION, re.VERBOSE)

# CLAUSE_PUNCT = r'^[.:;!?]$'
# CLAUSE_PUNCT_RE = re.compile(CLAUSE_PUNCT)

# # Happy and sad emoticons

# HAPPY = set([
#     ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
#     ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
#     '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
#     'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
#     '<3'
#     ])

# SAD = set([
#     ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
#     ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
#     ':c', ':{', '>:\\', ';('
#     ])


# def split_train_test(all_instances, n=None):
#     '''
#     Randomly split `n` instances of the dataset into train and test sets.

#     :param all_instances: a list of instances (e.g. documents) that will be split.
#     :param n: the number of instances to consider (in case we want to use only a
#         subset).
#     :return: two lists of instances. Train set is 8/10 of the total and test set
#         is 2/10 of the total.
#     '''
#     random.seed(12345)
#     random.shuffle(all_instances)
#     if not n or n > len(all_instances):
#         n = len(all_instances)
#     train_set = all_instances[:int(.8*n)]
#     test_set = all_instances[int(.8*n):n]

#     return train_set, test_set

# def save_file(content, filename):
#     """
#     Store `content` in `filename`. Can be used to store a SentimentAnalyzer.
#     """
#     print("Saving", filename)
#     with codecs.open(filename, 'wb') as storage_file:
#         # The protocol=2 parameter is for python2 compatibility
#         pickle.dump(content, storage_file, protocol=2)

# def mark_negation(document, double_neg_flip=False, shallow=False):
#     """
#     Append _NEG suffix to words that appear in the scope between a negation
#     and a punctuation mark.

#     :param document: a list of words/tokens, or a tuple (words, label).
#     :param shallow: if True, the method will modify the original document in place.
#     :param double_neg_flip: if True, double negation is considered affirmation
#         (we activate/deactivate negation scope everytime we find a negation).
#     :return: if `shallow == True` the method will modify the original document
#         and return it. If `shallow == False` the method will return a modified
#         document, leaving the original unmodified.

#     >>> sent = "I didn't like this movie . It was bad .".split()
#     >>> mark_negation(sent)
#     ['I', "didn't", 'like_NEG', 'this_NEG', 'movie_NEG', '.', 'It', 'was', 'bad', '.']
#     """
#     if not shallow:
#         document = deepcopy(document)
#     # check if the document is labeled. If so, do not consider the label.
#     labeled = document and isinstance(document[0], (tuple, list))
#     if labeled:
#         doc = document[0]
#     else:
#         doc = document
#     neg_scope = False
#     for i, word in enumerate(doc):
#         if NEGATION_RE.search(word):
#             if not neg_scope or (neg_scope and double_neg_flip):
#                 neg_scope = not neg_scope
#                 continue
#             else:
#                 doc[i] += '_NEG'
#         elif neg_scope and CLAUSE_PUNCT_RE.search(word):
#             neg_scope = not neg_scope
#         elif neg_scope and not CLAUSE_PUNCT_RE.search(word):
#             doc[i] += '_NEG'

#     return document

# def extract_unigram_feats(document, unigrams, handle_negation=False):
#     """
#     Populate a dictionary of unigram features, reflecting the presence/absence in
#     the document of each of the tokens in `unigrams`.

#     :param document: a list of words/tokens.
#     :param unigrams: a list of words/tokens whose presence/absence has to be
#         checked in `document`.
#     :param handle_negation: if `handle_negation == True` apply `mark_negation`
#         method to `document` before checking for unigram presence/absence.
#     :return: a dictionary of unigram features {unigram : boolean}.

#     >>> words = ['ice', 'police', 'riot']
#     >>> document = 'ice is melting due to global warming'.split()
#     >>> sorted(extract_unigram_feats(document, words).items())
#     [('contains(ice)', True), ('contains(police)', False), ('contains(riot)', False)]
#     """
#     features = {}
#     if handle_negation:
#         document = mark_negation(document)
#     for word in unigrams:
#         features['contains({0})'.format(word)] = word in set(document)
#     return features

# reload(sys)
# sys.setdefaultencoding('utf8')
# # sentim_analyzer = SentimentAnalyzer()
# sentiment_analyzer = pickle.load(open('sa_subjectivity.pickle'))
# word_tokenizer = regexp.WhitespaceTokenizer()
# tokenized_text = [word.lower() for word in word_tokenizer.tokenize('Spanish is a language')]
# feats = dict([(word, True) for word in tokenized_text])
# # subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')]
# # obj_docs = [(sent, 'obj') for sent in subjectivity.sents(categories='obj')]

# # train_subj_docs, test_subj_docs = split_train_test(subj_docs)
# # train_obj_docs, test_obj_docs = split_train_test(obj_docs)

# # training_docs = train_subj_docs+train_obj_docs
# # testing_docs = test_subj_docs+test_obj_docs

# # all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in training_docs])
# # unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)
# # sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)

# # training_set = sentim_analyzer.apply_features(training_docs)
# # test_set = sentim_analyzer.apply_features(testing_docs)

# # classifier = NaiveBayesClassifier.train(training_set)
# # save_file(classifier,'subjectivity_classifier.pickle')


# print sentiment_analyzer.prob_classify(feats)

# # from nltk.sentiment.vader import SentimentIntensityAnalyzer
# # sentences = ["VADER is smart, handsome, and funny.", # positive sentence example
# #     "VADER is smart, handsome, and funny!", # punctuation emphasis handled correctly (sentiment intensity adjusted)
# #     "VADER is very smart, handsome, and funny.",  # booster words handled correctly (sentiment intensity adjusted)
# #     "VADER is VERY SMART, handsome, and FUNNY.",  # emphasis for ALLCAPS handled
# #     "VADER is VERY SMART, handsome, and FUNNY!!!",# combination of signals - VADER appropriately adjusts intensity
# #     "VADER is VERY SMART, really handsome, and INCREDIBLY FUNNY!!!",# booster words & punctuation make this close to ceiling for score
# #     "The book was good.",   # positive sentence
# #     "The book was kind of good.", # qualified positive sentence is handled correctly (intensity adjusted)
# #     "The plot was good, but the characters are uncompelling and the dialog is not great.", # mixed negation sentence
# #     "A really bad, horrible book.",  # negative sentence with booster words
# #     "At least it isn't a horrible book.", # negated negative sentence with contraction
# #     ":) and :D",      # emoticons handled
# #     "",   # an empty string is correctly handled
# #     "Today sux",      #  negative slang handled
# #     "Today sux!",    #  negative slang with punctuation emphasis handled
# #     "Today SUX!",    #  negative slang with capitalization emphasis
# #     "Today kinda sux! But I'll get by, lol" # mixed sentiment example with slang and constrastive conjunction "but"
# # ]
# # paragraph = "It was one of the worst movies I've seen, despite good reviews. \
# #     Unbelievably bad acting!! Poor direction. VERY poor production. \
# #     The movie was bad. Very bad movie. VERY bad movie. VERY BAD movie. VERY BAD movie!"

# # from nltk import tokenize
# # line_list = tokenize.sent_tokenize(paragraph)
# # sentences.extend(line_list)
# # sid = SentimentIntensityAnalyzer()
# # for sentence in sentences:
# #     ss = sid.polarity_scores(sentence)
# #     print ss['neu']

# from nltk.sentiment import util
# from nltk.tokenize import regexp
# import sys, pickle

# reload(sys)
# sys.setdefaultencoding('utf8')
# # util.demo_sent_subjectivity('Chinese is a language')

# sentim_analyzer = pickle.load(open('sa_subjectivity.pickle'))
# word_tokenizer = regexp.WhitespaceTokenizer()
# tokenized_text = [word.lower() for word in word_tokenizer.tokenize('Spanish is a language')]
# instance_feats = sentim_analyzer.apply_features([tokenized_text], labeled=False)
# sentim_analyzer.classifier.prob_classify(instance_feats[0]).prob('subj')

# from nltk.corpus import wordnet as wn
# from itertools import chain

# synonyms = wn.synsets('video_game_addition')
# print synonyms[0]
# hypernyms = synonyms[0].hypernyms()
# hyponyms = synonyms[0].hyponyms()
# synonyms_list = list(set(chain.from_iterable([word.lemma_names() for word in synonyms])))
# hypernyms_list = list(set(chain.from_iterable(word.lemma_names() for word in hypernyms)))
# hyponyms_list = list(set(chain.from_iterable(word.lemma_names() for word in hyponyms)))
# print hypernyms

# from nltk.corpus import wordnet as wn

# boxing = wn.synsets('boxing')[0]
# contact_sport = wn.synsets('contact_sport')[0]

# print boxing.wup_similarity(contact_sport)
# import csv
# with open('eggs.csv', 'wb') as csvfile:
#     spamwriter = csv.writer(csvfile, delimiter=' ',
#                             quotechar=',', quoting=csv.QUOTE_MINIMAL)
#     spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])

# from nltk import ne_chunk, pos_tag, word_tokenize
# from nltk.tree import Tree
# def get_continuous_chunks(text):
#     chunked = ne_chunk(pos_tag(word_tokenize(text)))
#     print chunked
#     prev = None
#     continuous_chunk = []
#     current_chunk = []
#     for i in chunked:
#         if type(i) == Tree:
#             current_chunk.append(" ".join([token for token, pos in i.leaves()]))
#         elif current_chunk:
#             named_entity = " ".join(current_chunk)
#             if named_entity not in continuous_chunk:
#                 continuous_chunk.append(named_entity)
#                 current_chunk = []
#             else:
#                 continue
#     return continuous_chunk

# my_sent = "I think the first video is funny"
# print get_continuous_chunks(my_sent)

import re

message = 'Sure, I think this video is cool: https://www.youtube.com/watch?v=123456789 will you take?'
link = re.search("(?P<url>www.[^\s]+)", message).group("url")
print link