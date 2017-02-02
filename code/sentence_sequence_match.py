from sequential_pattern import sequential_pattern
import glob, string, sys, json, csv
from nltk.tokenize import sent_tokenize, regexp
# from step_one import Step_one
from util import match_sequences
from util import read_file

reload(sys)
sys.setdefaultencoding('utf8')

sp = sequential_pattern()
# step_one = Step_one()

# topics = step_one.topics
# files = step_one.files

csvfile = open('New_data/sequence_non_claim_all.csv','a+')
writer = csv.writer(csvfile, delimiter=',')

with open('New_data/all_sequence.txt','r') as f:
    sequences = json.load(f)

path = 'New_data/csv/sentence_component_non_claim_with_index.csv'
claims = read_file('New_data/non_claims_random.txt')

with open(path, 'rb') as f:
    spamreader = csv.reader(f, delimiter=',')
    for row in spamreader:
        index = row[14]
        claim = claims[int(index)]
        topic = claim[0]
        sentence = claim[1].replace('[REF]','')
        print sentence
        encoded_sentence = sp.encode_sentence(sentence,topic)
        seq_features = match_sequences(encoded_sentence,sequences)
        seq_features.append(0)
        writer.writerow(seq_features)
#     topic = row[0]
#     file = 'data/wiki12_articles/' + name
#     if file != current_file:
#         with open(file,'r') as f2:
#             text = f2.read()
#         text = unicode(text, errors='replace')
#         current_file = file
#     sentence_index = row[20]
#     sentences = sent_tokenize(text)
#     sentence = sentences[int(sentence_index)-1]
#     sentence = str(sentence).translate(None, string.punctuation).replace('REF','')
#     try:
#         encoded_sentence = sp.encode_sentence(sentence,topic)
#         seq_features = match_sequences(encoded_sentence,sequences)
#     except:
#         seq_features = [0, 0, 0, 0, 0, 0, 0]
#     seq_features.extend([name,sentence_index])
#     writer.writerow(seq_features)
#     print seq_features


# claims = read_file('New_data/non_claims.txt')
# for claim in claims:
#     if len(claim) < 2:
#         continue
#     topic = claim[0]
#     sentence = claim[1]
#     encoded_sentence = sp.encode_sentence(sentence,topic)
#     seq_features = match_sequences(encoded_sentence,sequences)
#     seq_features.append(0)
#     writer.writerow(seq_features)
    # print seq_features
