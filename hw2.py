
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.utils.data.dataloader import DataLoader
from collections import defaultdict
from torchtext.vocab import Vocab
from torch.utils.data.dataset import Dataset, TensorDataset
from pathlib import Path
from collections import Counter

from utils import get_vocabs, split, PosDataReader, PosDataset

data_dir = os.getcwd()
print("* Data directory - ", data_dir)

path_train = os.path.join(data_dir, "train.labeled")
path_test = os.path.join(data_dir, "test.labeled")

paths_list = [path_train, path_test]
word_dict, pos_dict = get_vocabs(paths_list)

train = PosDataset(word_dict, pos_dict, path_train, padding=False)
train_dataloader = DataLoader(train, shuffle=True)
test = PosDataset(word_dict, pos_dict, path_test, padding=False)
test_dataloader = DataLoader(test, shuffle=False)


print("* Number of Train Tagged Sentences ", len(train))
print("* Number of Test Tagged Sentences ",len(test))

# class DnnPosTagger(nn.Module):
#     def __init__(self, word_embeddings, hidden_dim, word_vocab_size, tag_vocab_size, word_embedding_dim):
#         super(DnnPosTagger, self).__init__()
#         emb_dim = word_embeddings.shape[1]
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.word_embedding = nn.Embedding(word_vocab_size, word_embedding_dim)
#         # self.word_embedding = nn.Embedding.from_pretrained(word_embeddings, freeze=False)
#         self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, num_layers=2, bidirectional=True, batch_first=False)
#         self.hidden2tag = nn.Linear(hidden_dim*2, tag_vocab_size)

        
#     def forward(self, word_idx_tensor):
#         embeds = self.word_embedding(word_idx_tensor.to(self.device))   # [batch_size, seq_length, emb_dim]      
#         lstm_out, _ = self.lstm(embeds.view(embeds.shape[1], 1, -1))    # [seq_length, batch_size, 2*hidden_dim]
#         tag_space = self.hidden2tag(lstm_out.view(embeds.shape[1], -1)) # [seq_length, tag_dim]
#         tag_scores = F.log_softmax(tag_space, dim=1)                    # [seq_length, tag_dim]
#         return tag_scores
    

# def evaluate():
#     acc = 0
#     with torch.no_grad():
#         for batch_idx, input_data in enumerate(test_dataloader):
            
#             words_idx_tensor, pos_idx_tensor, sentence_length = input_data  
#             tag_scores = model(words_idx_tensor)
#             tag_scores = tag_scores.unsqueeze(0).permute(0,2,1)
            
#             _, indices = torch.max(tag_scores, 1)
#             acc += torch.mean(torch.tensor(pos_idx_tensor.to("cpu") == indices.to("cpu"), dtype=torch.float))
#         acc = acc / len(test)
#     return acc

# #CUDA_LAUNCH_BLOCKING=1  

# EPOCHS = 15
# WORD_EMBEDDING_DIM = 100
# HIDDEN_DIM = 1000
# word_vocab_size = len(train.word_idx_mappings)
# tag_vocab_size = len(train.pos_idx_mappings)

# model = DnnPosTagger(train_dataloader.dataset.word_vectors, HIDDEN_DIM, word_vocab_size, tag_vocab_size, WORD_EMBEDDING_DIM)

# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:0" if use_cuda else "cpu")

# if use_cuda:
#     model.cuda()

# # Define the loss function as the Negative Log Likelihood loss (NLLLoss)
# loss_function = nn.NLLLoss()

# # We will be using a simple SGD optimizer to minimize the loss function
# optimizer = optim.Adam(model.parameters(), lr=0.01)
# acumulate_grad_steps = 50 # This is the actual batch_size, while we officially use batch_size=1

# # Training start
# print("Training Started")
# accuracy_list = []
# loss_list = []
# epochs = EPOCHS
# for epoch in range(epochs):
#     acc = 0 # to keep track of accuracy
#     printable_loss = 0 # To keep track of the loss value
#     i = 0
#     for batch_idx, input_data in enumerate(train_dataloader):
#         i += 1
#         words_idx_tensor, pos_idx_tensor, sentence_length = input_data
        
#         tag_scores = model(words_idx_tensor)
#         tag_scores = tag_scores.unsqueeze(0).permute(0,2,1)
#         #print("tag_scores shape -", tag_scores.shape)
#         #print("pos_idx_tensor shape -", pos_idx_tensor.shape)
#         loss = loss_function(tag_scores, pos_idx_tensor.to(device))
#         loss = loss / acumulate_grad_steps
#         loss.backward()

#         if i % acumulate_grad_steps == 0:
#             optimizer.step()
#             model.zero_grad()
#         printable_loss += loss.item()
#         _, indices = torch.max(tag_scores, 1)
#         # print("tag_scores shape-", tag_scores.shape)
#         # print("indices shape-", indices.shape)
#         # acc += indices.eq(pos_idx_tensor.view_as(indices)).mean().item()
#         acc += torch.mean(torch.tensor(pos_idx_tensor.to("cpu") == indices.to("cpu"), dtype=torch.float))
#     printable_loss = acumulate_grad_steps*(printable_loss / len(train))
#     acc = acc / len(train)
#     loss_list.append(float(printable_loss))
#     accuracy_list.append(float(acc))
#     test_acc = evaluate()
#     e_interval = i
#     print("Epoch {} Completed,\tLoss {}\tAccuracy: {}\t Test Accuracy: {}".format(epoch + 1, np.mean(loss_list[-e_interval:]), np.mean(accuracy_list[-e_interval:]), test_acc))
        


# import matplotlib.pyplot as plt

# plt.plot(accuracy_list, c="red", label ="Accuracy")
# plt.xlabel("Epochs")
# plt.ylabel("Value")
# plt.legend()
# plt.show()

# plt.plot(loss_list, c="blue", label ="Loss")
# plt.xlabel("Epochs")
# plt.ylabel("Value")
# plt.legend()
# plt.show()


# 
# In HW2 we will implement the Graph-based Dependency Parser presented by Kiperwasser and Goldberg. In their [paper](https://https://arxiv.org/pdf/1603.04351.pdf), Kiperwasser and Goldberg proposed a Bidirectional LSTM model to extract features from the words and POS tags of a sentence, and generate an adjacency matrix representing the dependency edge scores between every two nodes (words) in the sentence. This matrix is then used by the Chu-Liu-Edmonds algorithm to create a valid parse tree.\
# For further details, please read sections 3 (Our Approach), 5 (Graph-based Parser),  Implementation details and Hyperparameter tuning under section 6 (Experiments and Results) from the [paper](https://arxiv.org/pdf/1603.04351.pdf).\
# **Please note:** instead of using the max-margin loss function presented in section 5 of the paper, we shall minimize the negative log-likelihood loss function ($NLLLoss$) which you will implement in your code:
# $$\min_{\theta} NLLLoss(D;\theta) = \\ \min_{\theta}\sum_{(X^i,Y^i) \in D} \sum_{(h,m) \in Y^i} -\frac{1}{|Y^i|} \cdot log(P(S^i_{h,m}|X^i,\theta))$$
# $$P(S^i_{h,m}|X^i,\theta)=\frac{exp(S^i_{h,m})}{\sum_{j=1}^{|Y^i|} exp(S^i_{j,m})}=Softmax(S^i_{h,m})$$
# Where:
# 
# * $ D = \{(X^i,Y^i)\}_{i=1}^n $, Dataset consisting of $n$ (sentence, true tree) pairs. 
# * $X^i = \{x_0=ROOT,x_1,...,x_{k_i}\}$ is the full sequence of words in the sentence.
# * $Y^i=\{(h,m)\}$ is the set of all (head_index, modifier_index) edges in the **true** dependency tree of sentence $X^i$. $|Y^i|=k_i$.
# * $S^i \in R^{(k_i+1)^2}$ is the score matrix for all possible (head, modifier) edges in the dependency graph of sentence $X^i$.
# The cell $S^i_{h,m}$ refers to the score of $h$ being the head of $m$ in sentence $X^i$.
# * $\theta$ are the network's learned parameters
# 
# Below we give a basic structure for the network's code implementation.
# 

from chu_liu_edmonds import decode_mst


# class KiperwasserDependencyParser(nn.Module):
#     def __init__(self, *args):
#       super(KiperwasserDependencyParser, self).__init__()
#       self.word_embedding = # Implement embedding layer for words (can be new or pretrained - word2vec/glove)
#       self.pos_embedding = # Implement embedding layer for POS tags
#       self.hidden_dim = self.word_embedding.embedding_dim + self.pos_embedding.embedding_dim
#       self.encoder = # Implement BiLSTM module which is fed with word+pos embeddings and outputs hidden representations
#       self.edge_scorer = # Implement a sub-module to calculate the scores for all possible edges in sentence dependency graph
#       self.decoder = decode_mst # This is used to produce the maximum spannning tree during inference
#       self.loss_function = # Implement the loss function described above

#     def forward(self, sentence):
#       word_idx_tensor, pos_idx_tensor, true_tree_heads = sentence

#       # Pass word_idx and pos_idx through their embedding layers

#       # Concat both embedding outputs

#       # Get Bi-LSTM hidden representation for each word+pos in sentence

#       # Get score for each possible edge in the parsing graph, construct score matrix     
        
#       # Use Chu-Liu-Edmonds to get the predicted parse tree T' given the calculated score matrix

#       # Calculate the negative log likelihood loss described above
      
#       return loss, predicted_tree