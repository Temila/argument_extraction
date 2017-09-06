def get_label(sentence, claims):
    for claim in claims:
        if claim in sentence:
            return 1
    return 0