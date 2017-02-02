from tf_idf import TF_IDF
from util import read_file
import operator

claims = read_file('New_data/claims.txt')
none_claims = read_file('New_data/non_claims.txt')

document_1 = [x[1] for x in claims]

document_2 = []
for x in none_claims:
    if len(x) < 2:
        continue
    else:
        document_2.append(x[1])

tfidf = TF_IDF(document_1,document_2)
result = tfidf.get_tf_idf_score()
print sorted(result.items(), key = operator.itemgetter(1), reverse=True)