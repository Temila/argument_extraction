import re
from nltk.tokenize import sent_tokenize
from util import read_essay_txt_file, read_essay_ann_file, read_essay_ann_file_no_premise
# from sequential_pattern import sequential_pattern
from sentence_component import Sentence_component
from sematic_similarity import Sematic_Similarity

ss = Sematic_Similarity()
punctuation = '"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~'

def get_label(sentence, claims):
    for claim in claims:
        if claim in sentence:
            return 1
    return 0

def format_output(features):
    output = ''
    for feature in features:
        output = output + str(feature) + '#'
    output = output + '\n'
    return output

def generate_sub_sentence(sentence):
    min_size = 4
    r = re.compile(r'[{}]'.format(re.escape(punctuation)))
    chunck_by_punc = r.split(sentence)
    candidates = [x for x in chunck_by_punc if len(x.split(' ')) >= min_size]
    for can in candidates:
        if 'that' in can:
            candidates.extend(can.split('that'))
    candidates.append(sentence)
    return candidates

with open('New_data/csv/data_sub_2.csv', 'a') as output_file:
    output_file.write('#Mas#Ecs_1#Ecs_2#Ecs_3#Snp#Cnf_1#Cnf_2#Cnf_3#Cnf_4#Sub#Sen#Slc#index#Label_wp#Label_np')
    index = 0
    for i in range(1,403):
        topic, data = read_essay_txt_file(i)
        topic.replace('\'','')
        claims_wp = read_essay_ann_file(i)
        claims_np = read_essay_ann_file_no_premise(i)
        sentences = sent_tokenize(data)
        #encode sentence
        print 'processing essay {}, {} sentences found'.format(str(i),len(sentences))
        for sentence in sentences:
            index += 1
            label_wp = get_label(sentence, claims_wp)
            label_np = get_label(sentence, claims_np)
            subsentences = generate_sub_sentence(sentence)
            # print subsentences
            for subsentence in subsentences:
                try:
                    sc = Sentence_component(subsentence,topic,ss)
                    features = sc.extract_features()
                    features.append(index)
                    features.append(label_wp)
                    features.append(label_np)
                    output_file.write(format_output(features))
                except:
                    continue
            print 'sentence {} done'.format(index)
