from sequential_pattern import sequential_pattern
import glob, string, sys, json, csv
from nltk.tokenize import sent_tokenize, regexp
# from step_one import Step_one
from util import match_sequences

reload(sys)
sys.setdefaultencoding('utf8')

sp = sequential_pattern()
# step_one = Step_one()

# topics = step_one.topics
# files = step_one.files

csvfile = open('data/sequence_features_with_seq.csv','a+')
writer = csv.writer(csvfile, delimiter=',')

with open('data/seq.txt','r') as f:
    sequences = json.load(f)

path = 'data/sentence_component.csv'

with open(path, 'rb') as f:
    spamreader = csv.reader(f, delimiter=',')
    current_file = ''
    text = ''
    for row in spamreader:
        name = row[12]
        topic = row[0]
        file = 'data/wiki12_articles/' + name
        if file != current_file:
            with open(file,'r') as f2:
                text = f2.read()
            text = unicode(text, errors='replace')
            current_file = file
        sentence_index = row[13]
        sentences = sent_tokenize(text)
        sentence = sentences[int(sentence_index)-1]
        sentence = str(sentence).translate(None, string.punctuation).replace('REF','')
        try:
            encoded_sentence = sp.encode_sentence(sentence,topic)
            seq_features = match_sequences(encoded_sentence,sequences)
        except:
            seq_features = [0, 0, 0, 0, 0, 0, 0]
        seq_features.extend([name,sentence_index])
        writer.writerow(seq_features)
        print seq_features