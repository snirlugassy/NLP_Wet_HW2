# from google.colab import drive
import sys

import torch
from torch import nn, cuda, tensor, zeros, cat
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import torchtext
from collections import defaultdict, OrderedDict
from itertools import combinations
from chu_liu_edmonds import decode_mst
from datetime import datetime
from main import extract_sentences, get_vocab, ParsingDataset, DependencyParsingNetwork
from send_email import send_email


edges_count = 0
correct_edges_count = 0

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_file = sys.argv[1]
    else:
        print("Missing model file name!")
        # model_file = input("Enter model file name:")
        model_file = "21-06-25_20-55.model"

    device = 'cuda' if cuda.is_available() else 'cpu'
    print("Device = ", device)

    model = torch.load(model_file, map_location=device)

    train_path = "train.labeled"

    train_sentences = extract_sentences(train_path)
    train_dataset = ParsingDataset(train_sentences)

    loss_function = nn.NLLLoss()
    log_softmax = nn.LogSoftmax(dim=1)
    L = 0

    for i in range(len(train_dataset)):
        # print("Current sentence: ", i,"/",len(train_dataset))
        (tokens_vector, pos_vector), arcs = train_dataset[i]
        tokens_vector = tokens_vector.to(device)
        pos_vector = pos_vector.to(device)
        arcs = arcs.to(device)
        scores = model(tokens_vector, pos_vector)
        # loss = loss_function(scores, arcs)
        loss = loss_function(scores[1:,], arcs[1:])
        L += loss

        # _scores = np.zeros(scores.shape)
        # for i in range(1, len(arcs)):
        #     _scores[arcs[i]][i] = 100

        mst, _ = decode_mst(scores.detach().numpy(), scores.shape[0], has_labels=False)

        edges_count += scores.shape[0]
        correct_edges_count += sum(np.equal(mst, arcs))

    accuracy = correct_edges_count / edges_count
    print("Accuracy = ", accuracy)
