from util import read_file,extract_candidate_chunks,balance_data
import random
import pandas as pd
# from nltk.tokenize import sent_tokenize
# import glob,sys

# reload(sys)
# sys.setdefaultencoding('utf8')
# claims = read_file('New_data/claims.txt')
# topic_article = read_file('New_data/articles.txt')
# articles = glob.glob('New_data/articles/*')
# topic_original_text_dict = {}
# original_text_topic_dict = {}
# article_topic_dict = {}

# all_claim_sentences = [claim[2] for claim in claims]

# for claim in claims:
#     topic = claim[0]
#     original_text_topic_dict[claim[2]] = topic
#     if topic in topic_original_text_dict.keys():
#         topic_original_text_dict[topic].append(claim[2])
#     else:
#         topic_original_text_dict[topic] = [claim[2]]

# for ta in topic_article:
#     topic = ta[0]
#     file_name = 'New_data/articles/clean_' + ta[2] + '.txt'
#     if topic in article_topic_dict.keys():
#         article_topic_dict[topic].append(file_name)
#     else:
#         article_topic_dict[topic] = [file_name]

# with open('fix.txt','r') as f:
#     data = f.read()
# lines = data.split('\n')
# for line in lines:
#     sentence = line.split(' : ')[1]
#     topic = original_text_topic_dict[sentence]
#     articles = article_topic_dict[topic]
#     temp = ''
#     for article in articles:
#         with open(article,'r') as f:
#             text = f.read()
#         if sentence in text:
#             temp = article
#             break
#     print sentence + ' : ' + article
# non_cliam_sentences = read_file('New_data/non_claims.txt')
# with open('New_data/non_claims.txt','r') as f:
#     data = f.read().replace('[REF]','')
# non_cliam_sentences = data.split('\n')
# random.shuffle(non_cliam_sentences)
# result = '\n'.join(non_cliam_sentences)
# with open('New_data/non_claims_random.txt','w') as f:
#     f.write(result)
# print extract_candidate_chunks('This house believes that the sale of violent video games to minors should be banned')
# test = [1,2,3,4,5,6,7,8,9,10]
# print test[0:5]
# print test[5:10]

# data_df = pd.read_csv('New_data/csv/sequences_data _full.csv',header = 0)
# features = []
# for i in range(134):
#     features.append(str(i+1))
# label = ['Label']
# print balance_data(data_df,features,label)

a = [1,2,3,4]
print type(a[0])
print isinstance(a,list) and all(isinstance(x,str) for x in a)