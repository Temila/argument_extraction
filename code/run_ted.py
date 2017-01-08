from sentence_component import Sentence_component
from nltk.tokenize import sent_tokenize, regexp
from sematic_similarity import Sematic_Similarity
from util import extract_candidate_chunks
import csv, sys, glob, re

reload(sys)
sys.setdefaultencoding('utf8')
ss = Sematic_Similarity()
regex = re.compile('\(.+?\)')

csvfile = open('data/ted_component_fixed.csv','a+')
articles = glob.glob("data/ted_select/*")
writer = csv.writer(csvfile, delimiter=',')
articles_done = []
try:
    with open('ted_done.txt','r') as f:
        articles_done = f.read().split('\n')[:-1]
except:
    pass

for article in articles:
    name = article.replace('data/ted_select/','')
    topic_raw = name.replace('_',' ')
    topics = extract_candidate_chunks(topic_raw)
    topic = '.'.join(topics)
    print topic
    if name in articles_done:
        print '{} as already been parsed'.format(name)
        continue
    with open(article) as f:
        text = f.read()
    text = unicode(text, errors='replace')
    sentences = sent_tokenize(text)
    sentence_count = 0
    print 'processing: ' + name + ' sentences found: ' + str(len(sentences)) 
    for sentence in sentences:
        sentence = regex.sub('', sentence)
        # print sentence
        sentence_count += 1
        try:
            sc = Sentence_component(str(sentence),topic,ss)
            features = sc.extract_features()
            features.extend([name,sentence_count])
            writer.writerow(features)
            print 'Sentence {} done'.format(sentence_count)
        except:
            print 'warning'
            with open('ted_error_sentence.txt','a+') as f:
                f.write(sentence + '\n')
        # if sentence_count >= 10:
        #     break
    with open('article_done.txt', 'a+') as f:
        f.write(name + '\n')