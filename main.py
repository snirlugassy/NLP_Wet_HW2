# -*- coding: utf-8 -*-
"""NLP_Wet2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1b09TbwVyMQeD0v3luXddIscnZc-43st4
"""

# from google.colab import drive
import torch
import numpy as np
import torchtext
from collections import defaultdict, OrderedDict
from itertools import combinations
from chu_liu_edmonds import decode_mst

train_path = "train.labeled"
test_path = "test.labeled"

ROOT = "_R_"

def extract_sentences(file_path):
    sentences = []
    with open(file_path, 'r') as f:
        cur_sentence = [(ROOT, ROOT, -1)]
        for line in f:
            if line != '\n':
                splitted = line.split('\t')
                cur_sentence.append((splitted[1], splitted[3], splitted[6]))
            else:
                sentences.append(cur_sentence)
                cur_sentence = [(ROOT, ROOT, -1)]
    return sentences


def get_vocab(sentences):
    word_dict = defaultdict(int)
    pos_dict = defaultdict(int)
    for sentence in sentences:
        for word, pos in sentence:
            word_dict[word] += 1
            pos_dict[pos] += 1
    return word_dict, pos_dict


train_sentences = extract_sentences(train_path)
test_sentences = extract_sentences(test_path)

train_nohead = []
for s in train_sentences:
    _s = [(x[0], x[1]) for x in s]
    train_nohead.append(_s)

test_nohead = []
for s in test_sentences:
    _s = [(x[0], x[1]) for x in s]
    test_nohead.append(_s)

word_count, pos_count = get_vocab(train_nohead + test_nohead)
word_vocab_size, pos_vocab_size = len(word_count.keys()), len(pos_count.keys())

word_idx = dict()
idx_word = dict()

words_list = list(word_count.keys())

for i in range(len(words_list)):
    w = words_list[i]
    word_idx[w] = i
    idx_word[i] = w

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DependencyParsingNetwork(torch.nn.Module):
    def __init__(self, hidden_dim, word_vocab_size, pos_vocab_size, word_embedding_dim):
        super(DependencyParsingNetwork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.word_embedding = torch.nn.Embedding(word_vocab_size, word_embedding_dim)
        self.lstm = torch.nn.LSTM(input_size=word_embedding_dim, hidden_size=hidden_dim, num_layers=2, bidirectional=True, batch_first=False)
        self.mlp = torch.nn.Linear(2 * 2 * HIDDEN_DIM, 1, bias=True)
        self.tanh = torch.nn.Tanh()

    def forward(self, word_idx):
        x = self.word_embedding(word_idx)
        x = torch.tensor(x, requires_grad=True)
        x, (hn, cn) = self.lstm(x)
        x = x.squeeze(0)
        output = torch.zeros(x.shape[0], x.shape[0])
        for i, j in combinations(range(len(x)), 2):
            output[i][j] = self.tanh(self.mlp(torch.cat((x[i],x[j]))))
        # for i in range(len(x.split(1))):
        #     for j in range(len(x.split(1))):
        #         if i != j:
        #             x_i = x.split(1)[i].reshape(-1)
        #             x_j = x.split(1)[j].reshape(-1)
        #             output[i][j] = self.tanh(self.mlp(torch.cat((x_i,x_j))))
        return output


def vectorize_sentence(sentence, to_idx):
    idxs = [to_idx[w] for w in sentence]
    return torch.tensor([idxs])


# def loss(ground_truth, output):
HIDDEN_DIM = 2
WORD_EMBEDDING_DIM = 3
EPOCHS = 5

model = DependencyParsingNetwork(HIDDEN_DIM, word_vocab_size, pos_vocab_size, WORD_EMBEDDING_DIM).to(device)

# loss = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
log_softmax = torch.nn.LogSoftmax(1)

edge_count = 0
correct_predicted_edge = 0

for epoch in range(EPOCHS):
    for sentence in train_sentences:
        s = vectorize_sentence([_y[0] for _y in sentence], word_idx)
        s_len = len(sentence)
        arcs = [( int(sentence[i][2]), i ) for i in range(1,len(sentence))]
        x = model.forward(s)
        y = torch.zeros_like(x)
        for (h,m) in arcs:
            y[h][m] = -log_softmax(x)[h][m]
        # y = [( int(sentence[i][2]), i ) for i in range(1,len(sentence))]
        loss = torch.sum(y)
        model.zero_grad()
        loss.backward()
        optimizer.step()

        mst, _ = decode_mst(x.detach().numpy(), s_len, has_labels=False)
        true_mst = np.array([int(ss[2]) for ss in sentence])

        edge_count += s_len - 1
        correct_predicted_edge += sum(mst == true_mst) - 1
    print("Epoch = " , epoch , "/", EPOCHS)
    print("Accuracy = ", correct_predicted_edge / edge_count)