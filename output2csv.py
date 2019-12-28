# fairseq 'generate.py' outputs into csv file

import re
import csv
import sys

def getOutputs(text):
    with open(text) as data:
        lines = data.readlines()
        sentences = []
        outputs = []
        flag = 0
        for line in lines:
            if flag:
                words = line.split("\t")

                # 余計なもの [1,2,3,...] を出力しているとき
                if len(words)==1:
                    continue
                
                words = [re.sub(r"\n", "", word) for word in words]
                if line[0] == "H":
                    outputs.append(words[2])
                    output = words[2].split()
                    outputs.append(str(len(output)))
                    unk = output.count("<unk>") / len(output)
                    outputs.append(str(unk))
                if line[0] == 'H':
                    outputs.append(words[1])

                if line[:3] == "B''":
                    if float(words[1]) > 100:
                        words[1] = '100.00'
                    outputs.append(words[1])
                    flag = 0

            if line[0] == "S":
                flag = 1
                if len(outputs)>0:
                    sentences.append(outputs)
                    outputs = []
                line = re.sub(r'\n', '', line).split('\t')
                sent_id = line[0].split('-')[1]
                outputs.append(sent_id)
                outputs.append(line[1])
                outputs.append(str(len(line[1].split())))
        sentences.append(outputs)
    return sentences


def write(f_name, features, sentences):
    csv_name = f_name.split(".")[0] + ".csv"
    with open(csv_name, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(features)
        for sentence in sentences:
            writer.writerow(sentence)


if __name__ == "__main__":
    f_name = sys.argv
    f_name = text[1]
    sentences = getOutputs(text)
    features = ['id', 'source', 'len(source)', 'output', 'len(output)',
                'unknown', 'log-likelihood', 'bleu']
    write(f_name, features, sentences)

