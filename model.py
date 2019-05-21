# The model of NMT
# Seq2seq + Attention

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import time
import math 

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device}:{type(device)}")


################################################
# Loading data files
################################################

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0:"SOS", 1:"EOS"}
        self.n_words = 2

    def addSentence(self, sentence):
        for word in sentence.split(" "):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Unicode -> ASCII.
def unicode2Ascii(s):
    return "".join(
        c for c in unicodedata.normalize("NFD",s)
        if unicodedata.category(c) != "Mn"
    )

# Normalization.
# TODO need more(For Japanese)
def normalizeString(s):
    s = unicode2Ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

# To read the data.
# File -> lines, and split into pairs. 
def readLangs(lang1, lang2, data_place, reverse=False):
    print("Reading lines...")
    
    # Read the file and split into lines.
    lines = open(data_place, encoding="utf-8").read().strip().split("\n")

    # Split every line into pairs and normalize.
    #pairs = [[normalizeString(s) for s in l.split("\t")] for l in lines]
    pairs = [[s.lower().strip() for s in l.split("\t")] for l in lines]
    # Reverse pairs, make Lang instances.
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def filterPair(p):
    return len(p[0].split(" ")) < MAX_LENGTH and \
        len(p[1].split(" ")) < MAX_LENGTH 

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


# Full process for preparing the data.(Eng -> Other)
# If you translate Other -> Eng, you should add reverse. ("reverse=True")
def prepareData(lang1, lang2, data_place, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, data_place, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs



################################################
# Seq2Seq model
################################################

MAX_LENGTH = 20
#MAX_LENGTH = 10

# The Encoder
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# The Decoder
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# Attention
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


################################################
# Training
################################################

# Preparing training data
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

# Append EOS token
def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

# Return input and target tensor
def tensorsFromPair(pair, input_lang, output_lang):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)



def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden
    
    # TODO
    use_teacher_forcing = False
    #use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    if flag == 0:
    # flag 0: training, 1: validation
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

    return loss.item() / target_length


# Measure time and average loss.

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def trainIters(encoder, decoder, n_iters, print_every=1000, learning_rate=0.01):
    flag = 0
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    #plot_loss_total = 0  # Reset every plot_every

    #training_pairs = [tensorsFromPair(random.choice(pairs)) for i in range(n_iters)]
    #######################
    # changed   
    training_pairs = [tensorsFromPair(pair, input_lang, output_lang) for pair in pairs]
    n_iters = len(pairs)
    #######################
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        #plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))


################################################
# Validation
################################################

# validate each epoch
# code like evaluation
# calculate loss for each epoch

def validation(encoder, decoder, pairs_val, learning_rate=0.01):
    flag = 1
    val_iters = len(pairs_val)
    loss_total = 0.
    pairs_val = [tensorsFromPair(pair, input_lang_val, output_lang_val) for pair in pairs_val]
    
    #encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    #decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    for iter in range(1, val_iters+1):
        val_pair = pairs_val[iter - 1]
        input_tensor = val_pair[0]
        target_tensor = val_pair[1]
        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        loss_total += loss
    
    loss_ave = loss_total / val_iters
    
    return loss_ave



################################################
# Evaluation
################################################

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang_test, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            # topv: top value, topi: top index
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                #decoded_words.append(output_lang.index2word[topi.item()])
                
                ##################################
                # Original part: using top Value

                # Calculate variance top 5 and 3
                top5_value, top5_index = decoder_output.data.topk(5)
                #print(top5_value.size())

                # may be [~][topk] or [topk][~]
                #top5_mean = top5_value[0].mean()
                #top3_mean = top5_value[0][:3].mean()
                
                #top5_var = sum(abs(top5_value[0] - top5_mean)**2) / 5
                #top3_var = sum(abs(top5_value[0][:3] - top3_mean)**2) / 3
                
                top5_var = sum(abs(top5_value[0] - topv.item())**2) / 4
                top3_var = sum(abs(top5_value[0][:3] - topv.item())**2) / 2

                # May be outputted "word(value,top3,top5)"
                # TODO Changed! output_lang_test -> output_lang 
                output_wordAndValue = output_lang.index2word[topi.item()]
                #output_wordAndValue += (f"(P:{topv.item()},Vtop3:{top3_var.item()},Vtop5:{top5_var.item()})")
                output_wordAndValue += (f"(P:{topv.item()},Vtop3:{top3_var},Vtop5:{top5_var})")
                decoded_words.append(output_wordAndValue)

                ##################################
                

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(encoder, decoder, testData_place,  n=10):
    # evalate all
    n = len(pairs_test)
    
    for i in range(n):
        #pair = random.choice(pairs_test)
        pair = pairs_test[i]
        print(f"input> {pair[0]}")
        print(f"target= {pair[1]}")
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = '\n'.join(output_words)
        print(f"output<\n{output_sentence}")
        print('')



if __name__ == "__main__":
    hidden_size = 256
    teacher_forcing_ratio = 0.5
    epoch = 1000
    best_val_loss = None
    unupdate = 0

    #####################################
    # Data place
    #####################################
    #trainData_place = "/lab/aida/datasets/fra-eng/fra.txt"
    trainData_place = "/lab/aida/datasets/ASPEC_fixed/train-1_fixed.txt"
    
    #devData_place = "/lab/aida/datasets/fra-eng/fra.txt"
    devData_place = "/lab/aida/datasets/ASPEC_fixed/dev_fixed.txt"
    
    #testData_place = "/lab/aida/datasets/fra-eng/fra.txt"
    testData_place = "/lab/aida/datasets/ASPEC_fixed/test_fixed.txt"

   
    ####################################
    # Prepare Data
    ####################################
    #input_lang, output_lang, pairs = prepareData('eng', 'fra', trainData_place, True)
    input_lang, output_lang, pairs = prepareData('jap', 'eng', trainData_place, False)

    #input_lang_val, output_lang_val, pairs_val = prepareData("eng", "fra", devData_place, True)
    input_lang_val, output_lang_val, pairs_val = prepareData("jap", "eng", devData_place, False)
    
    #input_lang_test, output_lang_test, pairs_test = prepareData('eng', 'fra', testData_place, True)
    input_lang_test, output_lang_test, pairs_test = prepareData('jap', 'eng', testData_place, False)
    
    
    #####################################
    # Training part
    #####################################
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
    
    learning_rate = 0.01

    encoder_optimizer = optim.SGD(encoder1.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(attn_decoder1.parameters(), lr=learning_rate)
   
    for x in range(epoch):
        print("#"*30)
        print(f"train --{x}-- times")
        print("#"*30)

        trainIters(encoder1, attn_decoder1, 75000, print_every=5000)


        ######################################
        # Validation part
        ######################################
        val_loss = validation(encoder1, attn_decoder1, pairs_val)
        print(f"val_loss: {val_loss}")

        if not best_val_loss or val_loss < best_val_loss:
            # save model and val_loss
            best_val_loss = val_loss
            torch.save(encoder1.state_dict(), "ja-en_encoder.pth")
            torch.save(attn_decoder1.state_dict(), "ja-en_decoder.pth")
            unupdate = 0
        else:
            unupdate += 1
            if unupdate >= 3:
                break
    
    print("#"*30)
    print("train finished")
    print("#"*30)

    
    ######################################
    # Test part
    ######################################
    # load the least val_loss model
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size)
    encoder1.load_state_dict(torch.load("ja-en_encoder.pth")) 
    encoder1.to(device)
    encoder1 = encoder1.to(device)

    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1)
    attn_decoder1.load_state_dict(torch.load("ja-en_decoder.pth"))
    attn_decoder1.to(device)
    attn_decoder1 = attn_decoder1.to(device)

    evaluateRandomly(encoder1, attn_decoder1, testData_place)


