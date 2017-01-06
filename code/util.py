from prefixspan import prefixspan

def match_sequences(encoded_sentence, sequences):
    pfs = prefixspan([encoded_sentence])
    pfs.mining_sequence({},0)
    result = pfs.parse_results_unsorted()
    target = [x[0] for x in sequences]
    features = []
    length = len(encoded_sentence)
    for sequence in target:
        if sequence in result.keys():
            features.append(result[sequence] / length)
        else:
            features.append(0)
    return features