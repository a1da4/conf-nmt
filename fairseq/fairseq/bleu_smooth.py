# coding utf-8
# calculate bleu score of nmt results.
# n-gram: 1 <= n <= 4
# called by "model.py" or "main.py"
import math
from collections import defaultdict
from collections import Counter

#def splitAndNormalize(sentence):
    # normalize
    # split
    # use MeCab???


def ngram(sentence_split, n):
    # make n-gram and count
    # make key for ngram-dictionary
    ngram_list = []
    sentence_Length = len(sentence_split)
    for s in range(sentence_Length-n+1):
        ngram_list.append(" ".join(sentence_split[s:s+n]))
    
    ngramCounter = Counter(ngram_list)
    
    return ngramCounter


def ngram_totalCount(ngram_Dict):
    #totalCount = 1e-8
    totalCount = 0
    for key in ngram_Dict:
        totalCount += ngram_Dict[key]

    return totalCount


def ngram_matchCount(target_ngramDict, output_ngramDict):
    matchCount = 1
    for output_ngram in output_ngramDict:
        if output_ngram in target_ngramDict:
            matchCount += output_ngramDict[output_ngram]
    
    return matchCount


def precision_each(target_ngramDict, output_ngramDict):
    target_count = ngram_totalCount(target_ngramDict)
    output_count = ngram_totalCount(output_ngramDict)
    match_count = ngram_matchCount(target_ngramDict, output_ngramDict)
    # 0 で割ってしまうことを防ぐ。文長4未満への対策
    if target_count==0 or output_count==0:
        return 1
    precision_i = min(target_count, match_count) / output_count
    
    return precision_i


def precision_total(target_sentence_split, output_sentence_split):
    # n-gram overlap
    # calculate n(1~4)-gram precision.
    precision = 1
    for n in range(1, 4+1):
        #print(f"make {n}-gram")
        target_ngramDict = ngram(target_sentence_split, n) 
        output_ngramDict = ngram(output_sentence_split, n)
        #print(f"target:{target_ngramDict}\noutput:{output_ngramDict}")
        precision_i = precision_each(target_ngramDict, output_ngramDict)
        precision *= precision_i
        #print()
    precision = precision ** (1/4)
    #print("precision:{:.2f}".format(precision))
    
    return precision


def brevity_penalty(target_sentence_split, output_sentence_split):
    # brevity penalty
    # calculate min(1, exp(1 - reference-length/output-length))
    # must separated(use .split(" "))
    target_length = len(target_sentence_split)
    output_length = len(output_sentence_split)
    lengthRateEXP = math.exp(1 - target_length/output_length)
    brevity = min(1, lengthRateEXP)
    #print("brevity:{:.2f}".format(brevity))

    return min(1, lengthRateEXP)


def bleu(target_sentence, output_sentence):
    # calculate (brevity penalty) * (n-gram overlap)
    # use target and output sentence
    target_sentence = target_sentence.split(" ")
    output_sentence = output_sentence.split(" ")

    if len(target_sentence[-1])>1 and '.' in target_sentence[-1]:
        target_sentence = split_dot(target_sentence)
    if len(output_sentence[-1])>1 and '.' in output_sentence[-1]:
        output_sentence = split_dot(output_sentence)
    
    print(target_sentence, output_sentence)

    precision = precision_total(target_sentence, output_sentence)
    penalty = brevity_penalty(target_sentence, output_sentence)
    bleu_score = precision * penalty
    
    bleu_score *= 100

    return bleu_score


def split_dot(sentence):
    sentence[-1] = sentence[-1][:-1]
    sentence.append('.')
    
    return sentence

"""
if __name__ == "__main__":
    #target_sentence = "The NASA Opportunity rover is battling a massive dust storm on Mars ."
    #output_sentence = "The Opportunity rover is combatting a big sandstorm on Mars ."
    #output_sentence = "A NASA rover is fighting a massive storm on Mars ."
    #print(f"target:{target_sentence}\noutput:{output_sentence}")
    #print()
    bleu_score = bleu(target_sentence, output_sentence)
    
    #print("bleu_score:{:.2f}".format(bleu_score))
    bleu_score *= 100
    #print("bleu_score[%]:{:.0f}".format(bleu_score))
    return bleu_score
"""

