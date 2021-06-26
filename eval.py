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
from main import extract_sentences, get_vocab, ParsingDataset
from send_email import send_email


if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_file = sys.argv[1]
    else:
        print("Missing model file name!")
        # model_file = input("Enter model file name:")
        model_file = "21-06-25_20-51.model"

    device = 'cuda' if cuda.is_available() else 'cpu'
    print("Device = ", device)

    model = torch.load(model_file).to(device)

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
        arc = arcs.to(device)
        scores = model(tokens_vector, pos_vector)
        loss = loss_function(scores, arcs)
        L += loss
        mst, _ = decode_mst(scores.detach().numpy(), len(tokens_vector), has_labels=False)
        # true_mst = np.array([int(ss[2]) for ss in sentence])

        # edge_count += s_len - 1
        # correct_predicted_edge += sum(mst == true_mst) - 1
