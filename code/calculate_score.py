import glob
from util import read_essay_ann_file
from nltk.tokenize import sent_tokenize

name_questions = [
                    'focus of government budget',
                    'international tourism',
                    'living and studying overseas',
                    'newspapers',
                    'roommates'
                 ]

def run(folder_num):
    path = "New_data/TED_Annotation/Annotate_{}/".format(str(folder_num))
    files = glob.glob(path+'*.txt')
    for file in files:
        filename = get_file_name(file)
        if filename in name_questions:
            answers = get_answer(filename)
            return calculate(file, answers)

def get_answer(filename):
    path = 'New_data/TED_Annotation/Answers/{}.ann'.format(filename)
    with open(path, 'r') as f:
        return f.read().split('\n')

def get_file_name(file_path):
    return file_path[35:-6]

def calculate(file, answers):
    correct = 0.0
    with open(file, 'r') as f:
        annotates = sent_tokenize(f.read())
    for annotate in annotates:
        for answer in answers:
            if answer in annotate or answer is annotate:
                correct += 1
                break
    precision = correct / len(annotates)
    recall = correct / len(answers)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

for i in range(1,10):
    try:
        print 'annotation_{}'.format(str(i))
        precision, recall, f1 = run(i)
        print 'precision: {}'.format(str(precision))
        print 'recall: {}'.format(str(recall))
        print 'f1: {}'.format(str(f1))
    except:
        pass