# coding utf-8
# calculate bleu score of nmt results.
# n-gram: 1 <= n <= 4
# called by "model.py" or "main.py"

def ngram():
# n-gram overlap
# calculate n(1~4)-gram precision.


def penalty():
# brevity penalty
# calculate min(1, exp(1 - reference-length/output-length))

def bleu():
# calculate (brevity penalty) * (n-gram overlap)

if __name__ == "__main__":
    bleuScore = bleu()
    return bleuScore

