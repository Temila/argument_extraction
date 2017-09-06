import glob
import itertools
from nltk.tokenize import sent_tokenize

name_questions = [
                    'focus of government budget',
                    'international tourism',
                    'living and studying overseas',
                    'newspapers',
                    'roommates'
                 ]

def calculate(file_1, file_2):
    if validate(file_1, file_2):
        filename = get_file_name(file_1)
        ted_path = 'New_data/TED_Annotation/ted/{}.txt'.format(filename)
        with open(file_1, 'r') as ann_file_1:
            annotates_1 = sent_tokenize(ann_file_1.read())
        with open(file_2, 'r') as ann_file_2:
            annotates_2 = sent_tokenize(ann_file_2.read())
        with open(ted_path, 'r') as ted_file:
            ted = sent_tokenize(ted_file.read())
        false_1 = minus(ted, annotates_1)
        false_2 = minus(ted, annotates_2)
        m = float(len(set(annotates_1) & set(annotates_2)))
        n = float(len(set(false_1) & set(false_2)))
        x = float(len(set(annotates_2) & set(false_1)))
        y = float(len(set(annotates_1) & set(false_2)))
        T = m + n + x + y
        OA = (m + n) / T
        AC = (m + y) / T * (m + x) / T + (y + n) / T *(x + n) / T
        Kappa = (OA - AC) / (1 - AC)
        return Kappa, filename
    else:
        return None, None

def minus(list_A, list_B):
    return [item for item in list_A if item not in list_B]

def get_file_name(file_path):
    return file_path[35:-6]

def validate(file_1, file_2):
    filename_1 = get_file_name(file_1)
    filename_2 = get_file_name(file_2)
    def _is_same(file_1, file_2):
        return filename_1 == filename_2
    def _is_quality_control(file_1, file_2):
        return filename_1 not in name_questions and\
               filename_2 not in name_questions
    return _is_same(file_1, file_2) and _is_quality_control(file_1, file_2)

def get_foldername(file):
    return file.split('/')[2]

annotate_folders = glob.glob('New_data/TED_Annotation/Annotate_*/')
All_TEDS = []
for annotate_folder in annotate_folders:
    files = glob.glob(annotate_folder + '*.txt')
    teds = [x for x in files if x not in name_questions]
    All_TEDS.extend(teds)
combines = list(itertools.combinations(All_TEDS, 2))
for (file_1, file_2) in combines:
    result, filename = calculate(file_1, file_2)
    if result:
        folder_1 = get_foldername(file_1)
        folder_2 = get_foldername(file_2)
        print 'agreement between {} in {} and {} is: {}'\
                .format(filename, folder_1, folder_2, result) 
