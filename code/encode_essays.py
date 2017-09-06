from nltk.tokenize import sent_tokenize
from util import read_essay_txt_file, read_essay_ann_file
# from sequential_pattern import sequential_pattern
from sentence_component import Sentence_component
from sematic_similarity import Sematic_Similarity

ss = Sematic_Similarity()

def get_label(sentence, claims):
    for claim in claims:
        if claim in sentence:
            return 1
    return 0

def format_output(features, label):
    output = ''
    for feature in features:
        output = output + str(feature) + '#'
    output = output + str(label) + '\n'
    return output

with open('New_data/data_wp_2.csv', 'a') as output_file:
    for i in range(1,403):
        topic, data = read_essay_txt_file(i)
        claims = read_essay_ann_file(i)
        sentences = sent_tokenize(data)
        #encode sentence
        # headers = ['Mas','Ecs_1','Ecs_2','Ecs_3','Snp','Cnf_1','Cnf_2','Cnf_3','Cnf_4','Sub','Sen','Slc']
        print 'processing essay {}, {} sentences found'.format(str(i),len(sentences))
        index = 0
        for sentence in sentences:
            index += 1
            sc = Sentence_component(sentence,topic,ss)
            features = sc.extract_features()
            label = get_label(sentence, claims)
            features.append(label)
            output_file.write(format_output(features, label))
            print 'sentence {} done'.format(index)
