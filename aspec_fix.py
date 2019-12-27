# Fix datasets -> (lang1)\t(lang2)
import re
import sys
import MeCab

data_place = sys.argv[1]

mecab = MeCab.Tagger("-Owakati")

with open(data_place) as data:
    lines = data.readlines()
    for line in lines:
        # if "train-(1 | 2 | 3)"
        # (similarity score) ||| (Field of symbols - Document ID) ||| (Sentences ID) ||| (Japanese) ||| (English)
        #"""
        sentences = line.split(" ||| ")
        sentences[3] = mecab.parse(sentences[3])
        sentences[3] = re.sub("\n", "", sentences[3])
        bilingual = "\t".join(sentences[3:])
        print(bilingual,end="")
        """
        # else
        # (Field of symbols - Document ID) ||| (Sentences ID) ||| (Japanese) ||| (English)
        sentences = line.split(" ||| ")
        sentences[2] = mecab.parse(sentences[2])
        sentences[2] = re.sub("\n", "", sentences[2])
        bilingual = "\t".join(sentences[2:])
        print(bilingual,end="")
        """


