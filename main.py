# -*- coding: utf-8 -*-
"""NLP_Wet2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1b09TbwVyMQeD0v3luXddIscnZc-43st4
"""

# from google.colab import drive
import torch
from torch import nn, cuda, tensor, zeros, cat
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torchtext
from collections import defaultdict, OrderedDict
from itertools import combinations
from chu_liu_edmonds import decode_mst

train_path = "train.labeled"
test_path = "test.labeled"

ROOT = "_R_"

class ParsingDataset(Dataset):
    def __init__(self, sentences, word_idx, pos_idx):
        self.sentences = sentences
        self.word_idx = word_idx
        self.pos_idx = pos_idx

    def vectorize_tokens(self, tokens):
        idxs = [self.word_idx[w] for w in tokens]
        return tensor([idxs])

    def vectorize_pos(self, pos):
        pos_vector = zeros((len(pos), len(self.pos_idx.keys())))
        for i in range(len(pos)):
            pos_vector[(i, self.pos_idx[pos[i]])] = 1
        return pos_vector

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        s = self.sentences[idx]
        tokens_vector = self.vectorize_tokens([w[0] for w in s])
        pos_vector = self.vectorize_pos([w[1] for w in s])
        arcs = [(int(s[i][2]), i) for i in range(1, len(s))]
        return (tokens_vector, pos_vector), arcs

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

# Set word id from vocab
word_idx = dict()
idx_word = dict()
words_list = list(word_count.keys())
for i in range(len(words_list)):
    w = words_list[i]
    word_idx[w] = i
    idx_word[i] = w

# Set POS id from vocab
pos_idx = dict()
idx_pos = dict()
pos_list = list(pos_count.keys())
for i in range(len(pos_list)):
    pos = pos_list[i]
    pos_idx[pos] = i
    idx_pos[i] = pos


class DependencyParsingNetwork(nn.Module):
    def __init__(self, hidden_dim, word_vocab_size, word_embedding_dim, pos_vocab_size):
        super(DependencyParsingNetwork, self).__init__()
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.word_embedding = nn.Embedding(word_vocab_size, word_embedding_dim)
        # TODO: add dropout arg to lstm
        self.lstm = nn.LSTM(input_size=word_embedding_dim + pos_vocab_size, hidden_size=hidden_dim, num_layers=2, bidirectional=True, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(2 * 2 * HIDDEN_DIM, 1),
            nn.Tanh()
        )
        # self.mlp = nn.Linear(2 * 2 * HIDDEN_DIM, 1)
        # self.tanh = nn.Tanh()

    def forward(self, token_vector, pos_vector):
        x = cat((self.word_embedding(token_vector).squeeze(0), pos_vector) , dim=1)
        x = x.unsqueeze(0)
        x, (hn, cn) = self.lstm(x)
        x = x.squeeze(0)
        output = zeros(x.shape[0], x.shape[0])
        for _i, _j in combinations(range(len(x)), 2):
            output[_i][_j] = self.tanh(self.mlp(cat((x[_i],x[_j]))))
        # for i in range(x.shape[0]):
        #     for j in range(x.shape[0]):
        #         if i != j:
        #             x_i = x[i]
        #             x_j = x[j]
        #             output[i][j] = self.tanh(self.mlp(cat((x_i,x_j))))
        return output


def vectorize_tokens(tokens, to_idx):
    idxs = [to_idx[w] for w in tokens]
    return tensor([idxs])


def vectorize_pos(pos, to_idx):
    pos_vector = zeros((len(pos), len(to_idx.keys())))
    for i in range(len(pos)):
        pos_vector[(i, to_idx[pos[i]])] = 1
    return pos_vector


# def loss(ground_truth, output):
HIDDEN_DIM = 50
WORD_EMBEDDING_DIM = 300
EPOCHS = 10


train_dataset = ParsingDataset(train_sentences, word_idx, pos_idx)

device = 'cuda' if cuda.is_available() else 'cpu'
print("Device = ", device)

model = DependencyParsingNetwork(HIDDEN_DIM, word_vocab_size, WORD_EMBEDDING_DIM,  pos_vocab_size).to(device)

# loss = nn.NLLLoss()
optimizer = Adam(model.parameters(), lr=0.5)
log_softmax = nn.LogSoftmax(dim=1)

edge_count = 0
correct_predicted_edge = 0


for epoch in range(EPOCHS):
    L = 0
    for i in range(len(train_dataset)):
        (tokens_vector, pos_vector), arcs = train_dataset[i]

        # Reset gradients to zero
        model.zero_grad()

        # Forward
        x = model(tokens_vector, pos_vector)

        # Calculate scores for every possible arc
        y = torch.zeros_like(x)
        for (h,m) in arcs:
            y[h][m] = -log_softmax(x)[h][m]
        # y = [( int(sentence[i][2]), i ) for i in range(1,len(sentence))]

        # Calculate loss
        loss = torch.sum(y)
        L += float(loss)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # mst, _ = decode_mst(x.detach().numpy(), s_len, has_labels=False)
        # true_mst = np.array([int(ss[2]) for ss in sentence])
        #
        # edge_count += s_len - 1
        # correct_predicted_edge += sum(mst == true_mst) - 1
    print("Epoch = ", epoch, "/", EPOCHS)
    print("Loss = ", L)
    # print("Accuracy = ", correct_predicted_edge / edge_count)