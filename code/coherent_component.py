

class coherent_component:

    def __init__(self, sentence, length):
        self.sentence = sentence
        self.length = length

    def incomplete_quotes(self):
        quotes_num = self.sentence.count('"')
        return quotes_num % 2 == 1:

    def contrast_related_conjunctive(self):
        words = ['but','however', 'instead', 'on the contrary', 'on the other hand', 'in contrast', 'rather']
        for word in words:
            if word in self.sentence:
                return 1
                break
        return 0

    def segment_length(self):
        return self.length
        
    def unresolved_co_references(self):
        pass