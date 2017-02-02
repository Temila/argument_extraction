from util import read_file
from nltk.tokenize import sent_tokenize
import glob,sys

reload(sys)
sys.setdefaultencoding('utf8')
claims = read_file('New_data/claims.txt')
topic_article = read_file('New_data/articles.txt')
articles = glob.glob('New_data/articles/*')
topic_original_text_dict = {}
original_text_topic_dict = {}

article_topic_dict = {}

all_claim_sentences = [claim[2] for claim in claims]

for claim in claims:
    topic = claim[0]
    original_text_topic_dict[claim[2]] = topic
    if topic in topic_original_text_dict.keys():
        topic_original_text_dict[topic].append(claim[2])
    else:
        topic_original_text_dict[topic] = [claim[2]]

# print topic_original_text_dict['This house supports the one-child policy of the republic of China']
for ta in topic_article:
    topic = ta[0]
    file_name = 'New_data/articles/clean_' + ta[2] + '.txt'
    article_topic_dict[file_name] = topic

result = ''
has_topic = article_topic_dict.keys()
count_claim = 0
for article in articles:
    if article not in has_topic:
        continue
    with open(article, 'r') as f:
        text = f.read().decode('utf-8')
    sentences = sent_tokenize(text)
    topic = article_topic_dict[article]
    claim_sentences = topic_original_text_dict[topic]
    used_claim_sentences = []
    print len(claim_sentences)
    for sentence in sentences:
        is_claim = False
        for claim_sentence in claim_sentences:
            if claim_sentence in sentence or claim_sentence == sentence:
                try:
                    all_claim_sentences.remove(claim_sentence)
                except:
                    pass
                count_claim += 1
                is_claim = True
                break
        if not is_claim:
            result += topic + '<_break_>' + sentence + '\n'


with open('non_claims.txt','w') as f:
    f.write(result)

for s in all_claim_sentences:
    print original_text_topic_dict[s] + ' : ' + s