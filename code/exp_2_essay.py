from util import read_essay_txt_file, read_essay_ann_file, read_essay_ann_file_no_premise
from nltk.tokenize import sent_tokenize
import numpy as np

def read_result_from_tk():
    with open('tk_res', 'r') as f:
        r_list = f.read().split('\n')
    pred_norm = [ 1 if float(i) > 0 else 0 for i in r_list]
    return pred_norm

def get_tk_label_chunck(tk_labels, start, end):
    return tk_labels[start:end]

def get_label(sentence, claims):
    for claim in claims:
        if claim in sentence:
            return 1
    return 0

def label_for_essay_wp(file_number):
    topic, article = read_essay_txt_file(file_number)
    sents = sent_tokenize(article)
    claims = read_essay_ann_file(file_number)
    output = []
    for sent in sents:
        output.append(get_label(sent, claims))
    return output, len(sents)

def label_for_essay_np(file_number):
    topic, article = read_essay_txt_file(file_number)
    sents = sent_tokenize(article)
    claims = read_essay_ann_file_no_premise(file_number)
    output = []
    for sent in sents:
        output.append(get_label(sent, claims))
    return output

def calculate_percison(preds, labels):
    count = len(preds)
    a = 0.0
    b = 0.0
    for i in range(count):
        if preds[i] == 1:
            a += 1
            if labels[i] == 1:
                b += 1
    try:
        return float(b)/float(a)
    except:
        return 0

def calculate_recall(preds, labels):
    count = len(preds)
    a = 0.0
    b = 0.0
    for i in range(count):
        if labels[i] == 1:
            a += 1
            if preds[i] == 1:
                b += 1
    return float(b)/float(a)

def calculate_f1(p, r):
    if p == 0 and r == 0:
        return 0.0
    else:
        return 2*(p*r)/(p+r)

def run():
    start = 0
    end = 0
    tk_labels = read_result_from_tk()
    ps_wp = 0.0
    rs_wp = 0.0
    fs_wp = 0.0
    ps_np = 0.0
    rs_np = 0.0
    fs_np = 0.0
    for i in range(1,403):
        labels_wp, length = label_for_essay_wp(i)
        labels_np = label_for_essay_np(i)
        end += length
        tk_labels_chunck = get_tk_label_chunck(tk_labels, start, end)
        p_wp = calculate_percison(tk_labels_chunck,labels_wp)
        r_wp = calculate_recall(tk_labels_chunck,labels_wp)
        f_wp = calculate_f1(p_wp, r_wp)
        p_np = calculate_percison(tk_labels_chunck,labels_np)
        r_np = calculate_recall(tk_labels_chunck,labels_np)
        f_np = calculate_f1(p_np, r_np)
        ps_wp += p_wp/402
        rs_wp += r_wp/402
        fs_wp += f_wp/402
        ps_np += p_np/402
        rs_np += r_np/402
        fs_np += f_np/402
        start = end
    print '{:.2%},{:.2%},{:.2%}'.format(ps_wp,rs_wp,fs_wp)
    print '{:.2%},{:.2%},{:.2%}'.format(ps_np,rs_np,fs_np)

run()
