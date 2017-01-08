from pycorenlp import StanfordCoreNLP
import sys, pickle, string, json
from itertools import chain
from nltk.corpus import wordnet as wn
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment import util
from nltk.tokenize import regexp
from sequential_pattern import sequential_pattern
from util import match_sequences

class Sentence_component:

    def __init__(self,sentence,topic,ss):
        reload(sys)
        sys.setdefaultencoding('utf8')
        self.nlp = StanfordCoreNLP('http://localhost:9000')
        self.sid = SentimentIntensityAnalyzer()
        self.ss = ss
        self.topic = topic
        self.sentence = sentence
        self.sentence_output = self._parse_sentence(sentence)
        self.subjects = self._get_subjects()
        self.pos_tags = [x['pos'] for x in self.sentence_output['sentences'][0]['tokens']]
        self.named_entities = [x['ner'] for x in self.sentence_output['sentences'][0]['tokens']]
        with open('data/seq.txt','r') as f:
            self.sequences = json.load(f)

    def _parse_sentence(self, sentence):
        return self.nlp.annotate(sentence, properties={
            'annotators': 'tokenize,ssplit,pos,depparse,parse,ner',
            'outputFormat': 'json'
        })

    def _get_subjects(self):
        subjects = []
        subject_words = []
        nn_phases = []
        for x in self.sentence_output['sentences'][0]['enhancedPlusPlusDependencies']:
            if 'subj' in x['dep'] and x['dependentGloss'] not in subject_words:
                subject_words.append(x['dependentGloss'])
            if x['dep'] == 'compound':
                nn_phase = x['dependentGloss'] + ' ' + x['governorGloss']
                nn_phases.append(nn_phase)
        subjects.extend(subject_words)
        for nn_phase in nn_phases:
            for word in subject_words:
                if word in nn_phase and nn_phase not in subjects:
                    subjects.append(nn_phase)
                    break
        return subjects


    def _match_at_subject(self):
        '''
        Cosine Similarity between the topic and the subjects of the 
        candidate sentence
        '''
        MAS = [0.0]
        for subject in self.subjects:
            ws1 = subject.strip().split(' ')
            ws2 = self.topic.strip().split(' ')
            MAS.append(self.ss.n_similarity(ws1,ws2))
        return [max(MAS)]

    def _expanded_Cosine_Similarity(self):
        '''
        Cosine similarity between the Topic and semantic expansions of 
        the candidate sentence.
        '''
        sim_syno_list = [0.0]
        sim_hyper_list = [0.0]
        sim_hypo_list = [0.0]
        for subject in self.subjects:
            try:
                synonyms,hypernyms,hyponyms = self._find_expands(subject)
            except:
                synonyms = []
                hypernyms = []
                hyponyms = []
            max_syno = 0
            max_hyper = 0
            max_hypo = 0
            for x in synonyms:
                sim = self.ss.n_similarity(x.strip().split(' '), self.topic.strip().split(' '))
                if sim > max_syno:
                    max_syno = sim
            sim_syno_list.append(max_syno)
            for x in hypernyms:
                sim = self.ss.n_similarity(x.strip().split(' '), self.topic.strip().split(' '))
                if sim > max_hyper:
                    max_hyper = sim
            sim_hyper_list.append(max_hyper)
            for x in hyponyms:
                sim = self.ss.n_similarity(x.strip().split(' '), self.topic.strip().split(' '))
                if sim > max_hypo:
                    max_hypo = sim
            sim_hypo_list.append(max_hypo)
        return [max(sim_syno_list),max(sim_hyper_list),max(sim_hypo_list)]

    def _coreNLP_Feature(self):
        return [self._conjugate_that(), self._verb_in_present(), self._years(),self._location()]

    def _conjugate_that(self):
        if 'WHNP (WDT that)' in self.sentence_output['sentences'][0]['parse']:
            return 1.0
        else:
            return 0.0

    def _verb_in_present(self):
        if 'VBG' in self.pos_tags or 'VBP' in self.pos_tags or 'VBZ' in self.pos_tags:
            return 1.0
        else:
            return 0.0

    def _infinitive_verb(self):
        pass

    def _years(self):
        if 'DATE' in self.named_entities:
            return 1.0
        else:
            return 0.0

    def _location(self):
        if 'LOCATION' in self.named_entities:
            return 1
        else:
            return 0

    def _subjective_score(self):
        sentim_analyzer = pickle.load(open('sa_subjectivity.pickle'))
        word_tokenizer = regexp.WhitespaceTokenizer()
        tokenized_text = [word.lower() for word in word_tokenizer.tokenize(self.sentence)]
        instance_feats = sentim_analyzer.apply_features([tokenized_text], labeled=False)
        return [sentim_analyzer.classifier.prob_classify(instance_feats[0]).prob('subj')]

    def _sentiment_ratio(self):
        sentiment_score = self.sid.polarity_scores(self.sentence)
        return [1-sentiment_score['neu']]

    def _find_expands(self, subject):
        '''
        Use WordNet to expand subject
        '''
        new_word = subject.replace(' ','_').lower()
        synonyms = wn.synsets(new_word)
        hypernyms = synonyms[0].hypernyms()
        hyponyms = synonyms[0].hyponyms()
        synonyms_list = list(set(chain.from_iterable([word.lemma_names() for word in synonyms])))
        hypernyms_list = list(set(chain.from_iterable(word.lemma_names() for word in hypernyms)))
        hyponyms_list = list(set(chain.from_iterable(word.lemma_names() for word in hyponyms)))
        return [x.replace('_',' ') for x in synonyms_list],[x.replace('_',' ') for x in hypernyms_list],[x.replace('_',' ') for x in hyponyms_list]

    def _sequence_match(self):
        try:
            encoded_sentence = sp.encode_sentence(self.sentence,self.topic)
            seq_features = match_sequences(encoded_sentence,self.sequences)
        except:
            seq_features = [0, 0, 0, 0, 0, 0, 0]
        return seq_features

    def extract_features(self):
        features = [self.topic]
        MAS = self._match_at_subject()
        ECS = self._expanded_Cosine_Similarity()
        CNF = self._coreNLP_Feature()
        SUB = self._subjective_score()
        SEN = self._sentiment_ratio()
        # SEQ = self._sequence_match()
        features.extend(MAS)
        features.extend(ECS)
        features.extend(CNF)
        features.extend(SUB)
        features.extend(SEN)
        # features.extend(SEQ)
        return features


# from sematic_similarity import Sematic_Similarity

# ss = Sematic_Similarity()
# sc = Sentence_component('Video game addiction is excessive or compulsive use of computer and video games that interferes with daily life','video game',ss)