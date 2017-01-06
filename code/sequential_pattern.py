import sys, xlrd, glob, string, json, operator, random
from pycorenlp import StanfordCoreNLP
from nltk.tokenize import sent_tokenize, word_tokenize
from step_one import Step_one
from tf_idf import TF_IDF

class sequential_pattern:

    def __init__(self):
        reload(sys)
        sys.setdefaultencoding('utf8')
        self.nlp = StanfordCoreNLP('http://localhost:9000')
        with open('data/sentiment_words.txt','r') as f:
            self.sentiment_words = f.read()
        self.claim_sentences, self.None_claim_sentences = self.get_all_claim_sentences()
        # self.claim_words = self.get_claim_words()
        with open('data/claim_words.txt','r') as f:
            self.claim_words = json.load(f)
        # self.claim_words = [x[0] for x in self.claim_words]


    def get_all_claim_sentences(self):
        claim_sentences = {}
        None_claim_sentences = {}
        step_one = Step_one()
        topics = step_one.topics
        files = glob.glob("data/wiki12_articles/*")
        wb = xlrd.open_workbook('data/2014_7_18_ibm_CDCdata.xls')
        sheet_names = wb.sheet_names()
        sheet = wb.sheet_by_name(sheet_names[0])
        n_rows = sheet.nrows
        for file in files:
            with open(file) as f:
                text = f.read()
            text = unicode(text, errors='replace')
            sentences = sent_tokenize(text)
            name = file.replace('data/wiki12_articles/','')
            topic = topics[name]
            claim_sentences[name] = []
            None_claim_sentences[name] = []
            CDCs = step_one.get_CDCs_by_title(name)
            for CDC in CDCs:
                for sentence in sentences:
                    sentence = str(sentence).translate(None, string.punctuation).replace('REF','')
                    if CDC in sentence or CDC is sentence:
                        claim_sentences[name].append(sentence.lower())
                    else:
                        if len(sentence) > 50:
                            None_claim_sentences[name].append(sentence.lower())
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
        if word in topic:
            return 'topic'
        else:
            return None

    def is_claim_word(self, word):
        if word in self.claim_words:
            return 'claim'
        else:
            return None

    def is_sentiment_word(self, word, pos):
        if word in self.sentiment_words:
            return 'sentiment'
        else:
            return None

    def get_pos_tags(self, sentence):       
        sentence_output = self.nlp.annotate(str(sentence), properties={'annotators': 'pos','outputFormat': 'json'})
        pos = [x['pos'] for x in sentence_output['sentences'][0]['tokens']]
        for index, p in enumerate(pos):
            if p == 'NNS':
                pos[index] = 'NN'
        return pos

    def encode_CDC_words(self):
        output = []
        for topic in self.claim_sentences.keys():
            sentences = self.claim_sentences[topic]
            print 'encoding {}, {} sentences found'.format(topic, len(sentences))
            for sentence in sentences:
                pos_tags = self.get_pos_tags(sentence)
                words = word_tokenize(sentence)
                encoded_sentence = []
                for index, word in enumerate(words):
                    encoded_sentence.append([word,pos_tags[index],self.is_sentiment_word(word, pos_tags[index]),self.is_claim_word(word),self.in_topic(topic.lower(), word)])
                output.append(encoded_sentence)
        return output

    def encode_None_CDC_words(self):
        output = []
        for topic in self.None_claim_sentences.keys():
            sentences = self.None_claim_sentences[topic]
            length = len(sentences)
            checked_index = []
            print 'encoding {}, {} sentences found'.format(topic, length)
            for i in range(min(20,length)):
                index = random.randint(0,length-1)
                while index in checked_index:
                    index = random.randint(0,length-1)
                checked_index.append(index)
                sentence = sentences[index]
                words = word_tokenize(sentence)
                pos_tags = self.get_pos_tags(sentence)
                encoded_sentence = []
                if len(words) != len(pos_tags):
                    continue
                for index, word in enumerate(words):
                    encoded_sentence.append([word,pos_tags[index],self.is_sentiment_word(word, pos_tags[index]),self.is_claim_word(word),self.in_topic(topic.lower(), word)])
                output.append(encoded_sentence)
                # print encoded_sentence
        return output

    def encode_sentence(self, sentence, topic):
        pos_tags = self.get_pos_tags(sentence)
        words = word_tokenize(sentence)
        encoded_sentence = []
        if len(words) != len(pos_tags):
            return encoded_sentence
        for index, word in enumerate(words):
            encoded_sentence.append([word,pos_tags[index],self.is_sentiment_word(word, pos_tags[index]),self.is_claim_word(word),self.in_topic(topic.lower(), word)])
        return encoded_sentence


# sp = sequential_pattern()
# data = sp.encode_None_CDC_words()
# # data = sp.encode_CDC_words()
# with open('data/encoded_sentence_none_2.txt','w') as f:
#     json.dump(data, f)

# data_2 = sp.encode_CDC_words()
# with open('data/encoded_sentence_2.txt','w') as f:
#     json.dump(data_2, f)