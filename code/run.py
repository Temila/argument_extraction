# from step_one import Step_one
from sentence_component import Sentence_component
from nltk.tokenize import sent_tokenize
from sematic_similarity import Sematic_Similarity
from util import read_file
import csv, sys, threading

reload(sys)
sys.setdefaultencoding('utf8')
claims = read_file('New_data/non_claims.txt')
ss = Sematic_Similarity()

sc = Sentence_component('Violent video games can increase children\'s aggression','The sale of violent video games to minors should be banned',ss)
features = sc.extract_features()
print features

# csvfile = open('New_data/sentence_component_non_with_index.csv','a+')
# writer = csv.writer(csvfile, delimiter=',')

# def constituency_parse_tree(sentence):
#     return " ".join(nlp.annotate(sentence, properties={
#         'annotators': 'parse',
#         'outputFormat': 'json'
#     })['sentences'][0]['parse'].replace('\n','').split())

# def question_detect(sentence):
#     try:
#         tree = constituency_parse_tree(sentence)
#         if 'WHNP' in tree or 'WHADVP' in tree or'WHADJP' in tree:
#             if 'SQ' in tree and '?' in tree and len(word_tokenize(sentence)) > 3:
#                 print sentence
#                 count += 1
#     except:
#         pass

# count = 0
# for claim in claims:
#     count += 1
#     sentence_number = count
#     # print 'processing sentence {}'.format(str(sentence_number))
#     if not len(claim) == 2:
#         # print 'error occurred! Topic or Sentence is miss'
#         continue
#     topic = claim[0]
#     sentence = claim[1]
#     try:
#         question_detect(sentence)
#         # features.append(count-1)
#         # writer.writerow(features)
#     except:
#         # print 'error occurred! Fail to extract features'
#         continue
#     # print 'Done'
# with open('article_done.txt','r') as f:

#     articles_done = f.read().split('\n')[:-1]

# for article in articles:
#     name = article.replace('data/wiki12_articles/','')
#     if name in articles_done:
#         print '{} as already been parsed'.format(name)
#         continue
#     CDCs = step_one.get_CDCs_by_title(name)
#     topic = topics[name]
#     with open(article) as f:
#         text = f.read()
#     text = unicode(text, errors='replace')
#     sentences = sent_tokenize(text)
#     sentence_count = 0
#     print 'processing: ' + name + ' sentences found: ' + str(len(sentences)) 
#     for sentence in sentences:
#         sentence_count += 1
#         label = 0
#         for CDC in CDCs:
#             if CDC in sentence:
#                 label = 1
#                 CDCs.remove(CDC)
#                 break
#         # sentence = str(sentence).translate(None, string.punctuation).replace('REF','')
#         # print sentence
#         # sc = Sentence_component(str(sentence),topic,ss)
#         # features = sc.extract_features()
#         # features.extend([label,name,sentence_count])
#         # writer.writerow(features)
#         # print 'Sentence {} done'.format(sentence_count)
#         try:
#             sc = Sentence_component(str(sentence),topic,ss)
#             features = sc.extract_features()
#             features.extend([label,name,sentence_count])
#             writer.writerow(features)
#             print 'Sentence {} done'.format(sentence_count)
#         except:
#             print 'warning'
#             with open('error_sentence.txt','a+') as f:
#                 f.write(sentence + '\n')
#     with open('article_done.txt', 'a+') as f:
#         f.write(name + '\n')
#     if len(CDCs) > 0:
#         with open('wrong_label.txt','a+') as f:
#             for c in CDCs:
#                 f.write('{}: {}\n'.format(name,c))