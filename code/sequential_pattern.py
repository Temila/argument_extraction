import sys, xlrd, glob, string, json, operator, random
from pycorenlp import StanfordCoreNLP
from nltk.tokenize import sent_tokenize, word_tokenize
from step_one import Step_one
from tf_idf import TF_IDF
from util import read_file, extract_candidate_chunks

class sequential_pattern:

    def __init__(self):
        reload(sys)
        sys.setdefaultencoding('utf8')
        self.nlp = StanfordCoreNLP('http://localhost:9000')
        with open('data/sentiment_words.txt','r') as f:
            self.sentiment_words = f.read()
        self.claim_sentences, self.None_claim_sentences = self.get_all_sentences()
        # self.claim_words = self.get_claim_words()
        with open('data/claim_words.txt','r') as f:
            self.claim_words = json.load(f)
        # self.claim_words = [x[0] for x in self.claim_words]


    # def get_all_claim_sentences(self):
    #     claim_sentences = {}
    #     None_claim_sentences = {}
    #     step_one = Step_one()
    #     topics = step_one.topics
    #     files = glob.glob("data/wiki12_articles/*")
    #     wb = xlrd.open_workbook('data/2014_7_18_ibm_CDCdata.xls')
    #     sheet_names = wb.sheet_names()
    #     sheet = wb.sheet_by_name(sheet_names[0])
    #     n_rows = sheet.nrows
    #     for file in files:
    #         with open(file) as f:
    #             text = f.read()
    #         text = unicode(text, errors='replace')
    #         sentences = sent_tokenize(text)
    #         name = file.replace('data/wiki12_articles/','')
    #         topic = topics[name]
    #         claim_sentences[name] = []
    #         None_claim_sentences[name] = []
    #         CDCs = step_one.get_CDCs_by_title(name)
    #         for sentence in sentences:
# tfidf = TF_IDF(['this is sentence','this is also sentence','last sentence'],['doc','doc','doc'])
# print tfidf.get_tf_idf_score()
    #             is_cdc = False
    #             for CDC in CDCs:
    #                 if CDC in sentence:
    #                     claim_sentences[name].append(str(CDC).lower())
    #                     is_cdc = True
    #                     break
    #             if not is_cdc:
    #                 sentence = str(sentence).translate(None, string.punctuation).replace('REF','')
    #                 if len(sentence) > 50:
    #                     None_claim_sentences[name].append(sentence.lower())
    #     return claim_sentences, None_claim_sentences
    def get_all_sentences(self):
        claim_sentences = read_file('New_data/claims.txt', cut=True)
        None_claim_sentences = read_file('New_data/non_claims_random.txt')
        return claim_sentences, None_claim_sentences


    def get_claim_words(self):
        all_doc_dicts = {}
        for topic in self.claim_sentences:
            tfidf = TF_IDF(self.claim_sentences[topic],self.None_claim_sentences[topic])
            tf_idf_dict = tfidf.get_tf_idf_score()
            for word in tf_idf_dict:
                pos = self.nlp.annotate(word, properties={'annotators': 'pos','outputFormat': 'json'})['sentences'][0]['tokens'][0]['pos']
                if 'NN' in pos:
                    continue
                elif word in all_doc_dicts:
                    all_doc_dicts[word] += tf_idf_dict[word]
                else:
                    all_doc_dicts[word] = tf_idf_dict[word]
            print '{} done'.format(topic)
        return sorted(all_doc_dicts.items(), key=operator.itemgetter(1), reverse = True)


    def in_topic(self, topic, word):
        if isinstance(topic,list) and all(isinstance(x,str) for x in topic):
            topic_chunks = topic
        elif isinstance(topic,str):
            topic_chunks = extract_candidate_chunks(topic.lower())[1:]
            # if topic_chunks == []:
            #     topic_chunks = ['gambling']
        else:
            return None
        if word in topic_chunks:
            return 'topic'
        else:
            return None

    def is_claim_word(self, word):
        if word in self.claim_words:
            return 'claim'
        else:
            return None

    def is_sentiment_word(self, word, pos):
        try:
            if word in self.sentiment_words:
                return 'sentiment'
            else:
                return None
        except:
            return None

    def get_pos_tags(self, sentence):       
        sentence_output = self.nlp.annotate(str(sentence), properties={'annotators': 'pos','outputFormat': 'json'})
        sentences = sentence_output['sentences']
        pos = []
        words = []
        for sentence in sentences:
            pos.extend([x['pos'] for x in sentence['tokens']])
            words.extend([x['word'] for x in sentence['tokens']])
            for index, p in enumerate(pos):
                if p == 'NNS':
                    pos[index] = 'NN'
        return pos, words

    def encode_CDC_words(self):
        output = []
        for claim_sentence in self.claim_sentences:
            topic = claim_sentence[0]
            sentence = claim_sentence[1]
            print sentence
            # print 'encoding {}, {} sentences found'.format(topic, len(sentences))
            pos_tags, words = self.get_pos_tags(sentence)
            # words = word_tokenize(sentence)
            encoded_sentence = []
            for index, word in enumerate(words):
                encoded_sentence.append([word,pos_tags[index],self.is_sentiment_word(word, pos_tags[index]),self.is_claim_word(word),self.in_topic(topic, word)])
            output.append(encoded_sentence)
        return output

    def encode_None_CDC_words(self):
        output = []
        random.shuffle(self.None_claim_sentences)
        for non_claim_sentence in self.None_claim_sentences:
            if len(non_claim_sentence) < 2:
                continue
            topic = non_claim_sentence[0]
            sentence = non_claim_sentence[1]
            print sentence
            pos_tags,words = self.get_pos_tags(sentence)
            encoded_sentence = []
            if len(words) != len(pos_tags):
                continue
            for index, word in enumerate(words):
                encoded_sentence.append([word,pos_tags[index],self.is_sentiment_word(word, pos_tags[index]),self.is_claim_word(word),self.in_topic(topic, word)])
            output.append(encoded_sentence)
        return output

    def encode_sentence(self, sentence, topic):
        pos_tags, words = self.get_pos_tags(sentence)
        encoded_sentence = []
        if len(words) != len(pos_tags):
            return encoded_sentence
        for index, word in enumerate(words):
            encoded_sentence.append([word,pos_tags[index],self.is_sentiment_word(word, pos_tags[index]),self.is_claim_word(word),self.in_topic(topic, word)])
        return encoded_sentence


sp = sequential_pattern()
data = sp.encode_sentence('Violent video games can increase children\'s aggression','The sale of violent video games to minors should be banned')
print data