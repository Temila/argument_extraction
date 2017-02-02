import json, operator

with open('New_data/sequence_claim.txt') as f:
    sc = json.load(f)
sc_dict = {}
for n in sc:
    sc_dict[n[0]] = n[1]


with open('New_data/sequence_none_claim.txt') as f:
    snc = json.load(f)

snc_dict = {}
for n in snc:
    snc_dict[n[0]] = n[1]

for n in sc_dict.keys():
    if n in snc_dict:
        sc_dict[n] = sc_dict[n] - snc_dict[n]

result = sorted(sc_dict.items(), key=operator.itemgetter(1), reverse = True)

with open('New_data/seq.txt','w') as f:
    json.dump(result,f)
    
all_sequence = list(set(sc_dict.keys()).union(set(snc_dict.keys())))
with open('New_data/all_sequence.txt','w') as f:
    json.dump(all_sequence,f)

