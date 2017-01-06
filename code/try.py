# import pandas as pd
# data_df = pd.read_csv('data.csv',header = 0)
# true_label = data_df.loc[data_df['Label'] == 1]
# for i in range(35):
#     data_df = data_df.append(true_label)
# print data_df.loc[data_df['Label'] == 0]
# s = []
# print s == []
# s = ['this','is','a','test','sentnce']
# for i, x in enumerate(s):
#     print i
#     print s[i:]
import json, operator

with open('data/sequence_claim_2.txt') as f:
    sc = json.load(f)
sc_dict = {}
for n in sc:
    sc_dict[n[0]] = n[1]


with open('data/sequence_none_claim_2.txt') as f:
    snc = json.load(f)

snc_dict = {}
for n in snc:
    snc_dict[n[0]] = n[1]

for n in sc_dict.keys():
    if n in snc_dict:
        sc_dict[n] = sc_dict[n] / snc_dict[n]

result = sorted(sc_dict.items(), key=operator.itemgetter(1), reverse = True)

with open('data/seq_2.txt','w') as f:
    json.dump(result,f)
    