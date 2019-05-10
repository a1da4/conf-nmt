# coding utf-8
# calculate bleu score of nmt results.
# n-gram: 1 <= n <= 4
# called by "model.py" or "main.py"
import math
from collections import defaultdict

def splitAndNormalize(sentence):
    # normalize
    # split


def ngram(sentence, n):
    # make n-gram and count
    
    # make key for ngram-dictionary
    ngram_key = []
    sentence_Length = len(sentence)
    for s in range(sentence_Length-n+1):
        ngram_key.append(sentence[s:s+n])
    
    # make dictionary["ngram":"count"]
    ngramAndCount = defaultdict(int)
    for key in ngram_key:                                                                                                                                                                               
        ngramAndCount[key] += 1
     
    return ngramAndCount


def ngram_precision():
    # n-gram overlap
    # calculate n(1~4)-gram precision.
    
    
def brevity_penalty(target_sentence_split, output_sentence_split):
    # brevity penalty
    # calculate min(1, exp(1 - reference-length/output-length))
    # must separated(use .split(" "))
    target_length = len(target_sentence_split)
    output_length = len(output_sentence_split)
    lengthRateEXP = math.exp(1 - target_length/output_length)
    return min(1, lengthRateEXP)


def bleu(target_sentence, output_sentence):
    # calculate (brevity penalty) * (n-gram overlap)
    # use target and output sentence



if __name__ == "__main__":
    bleuScore = bleu(target_sentence, output_sentence)
    return bleuScore

