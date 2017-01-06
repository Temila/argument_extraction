import csv

path = 'data/sentence_component.csv'

with open(path, 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        print row[12]