# text data -> excel
# S, T, H, P-(id)"\t"

import re
import csv
import sys


text = sys.argv
text = text[1]

def getOutputs(text):
    with open(text) as data:
        lines = data.readlines()
        sentences = []
        outputs = []
        flag = 0
        for line in lines:
            if flag:
                lines = line.split("\t")
                if line[0] == "H":
                    outputs.append(lines[2])
                outputs.append(lines[1])
                if line[1] == "'" and line[2] == "'":
                    flag = 0

            if line[0] == "S":
                flag = 1
                #print(outputs)
                sentences.append(outputs)
                outputs = []
                outputs.append(line.split("\t")[1])
    return sentences


def write(sentences):
    f_name = text + ".csv"
    with open(f_name, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        for sentence in sentences:
            writer.writerow(sentence)

sentences = getOutputs(text)
write(sentences)

