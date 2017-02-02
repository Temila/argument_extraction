from sequential_pattern import sequential_pattern
import glob, string, sys, json, csv, re
from nltk.tokenize import sent_tokenize
from util import match_sequences
from util import extract_candidate_chunks

reload(sys)
sys.setdefaultencoding('utf8')

sp = sequential_pattern()
# step_one = Step_one()

# topics = step_one.topics
# files = step_one.files
regex = re.compile('\(.+?\)')
csvfile = open('data/ted_with_seq_fixed.csv','a+')
writer = csv.writer(csvfile, delimiter=',')

with open('data/seq.txt','r') as f:
    sequences = json.load(f)

path = 'data/ted_component_fixed.csv'

files = glob.glob("data/ted_select/*")
# for file in files:
#     with open(file, 'r') as f:
#         text = f.read()
#     text = unicode(text, errors='replace')
#     sentences = sent_tokenize(text)
#     sentence_count = 0
#     name = file.replace('data/ted_select/','')
#     topic_raw = name.replace('_',' ')
#     topics = extract_candidate_chunks(topic_raw)
#     topic = ','.join(topics)
#     if topic == '':
#         continue
#         print '{} has no topic!!!!!!!!!!'.format(name)
#     for sentence in sentences:
#         sentence_count += 1
#         sentence = regex.sub('', sentence)
#         sentence = str(sentence).translate(None, string.punctuation)
#         try:
#             encoded_sentence = sp.encode_sentence(sentence,topic)
#             seq_features = match_sequences(encoded_sentence,sequences)
#         except:
#             seq_features = [0, 0, 0, 0, 0, 0, 0]
#         seq_features.extend([name,sentence_count])
#         writer.writerow(seq_features)
#         print seq_features
with open(path, 'rb') as f:
    spamreader = csv.reader(f, delimiter=',')
    current_file = ''
    text = ''
    for row in spamreader:
        name = row[11]
        topic = row[0]
        file = 'data/ted_select/' + name
        if file != current_file:
            print file
            with open(file,'r') as f2:
                text = f2.read()
            text = unicode(text, errors='replace')
            current_file = file
        sentence_index = row[12]
        sentences = sent_tokenize(text)
        sentence = sentences[int(sentence_index)-1]
        sentence = regex.sub('', sentence)
        sentence = str(sentence).translate(None, string.punctuation)
        encoded_sentence = sp.encode_sentence(sentence,topic)
        seq_features = match_sequences(encoded_sentence,sequences)
        seq_features.extend([name,sentence_index])
        writer.writerow(seq_features)
        # print seq_features