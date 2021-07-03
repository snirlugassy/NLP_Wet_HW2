import math
from itertools import permutations
from collections import defaultdict, OrderedDict
from datetime import datetime
from time import time

import numpy as np
import torchtext
import torch
from torch import nn, cuda, tensor, zeros, cat
from torch.optim import Adam, Adagrad
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from advanced import DependencyParsingNetwork, extract_sentences, train_dataset, test_dataset, model_file_name
from chu_liu_edmonds import decode_mst

from parameters import TRIM_TRAIN_DATASET
from parameters import HIDDEN_DIM
from parameters import WORD_EMBEDDING_DIM
from parameters import POS_EMBEDDING_DIM
from parameters import EPOCHS
from parameters import BATCH_SIZE
from parameters import LEARNING_RATE
from parameters import TEST_SAMPLE_SIZE
from parameters import ROOT

comp_data_path = "comp.unlabeled"
comp_output_path = "comp_m1_206312506.labeled"


if __name__ == "__main__":
    sentences = extract_sentences(comp_data_path)
    device = 'cuda' if cuda.is_available() else 'cpu'
    print("Device = ", device)
    model = DependencyParsingNetwork(train_dataset.pos_vocab_size, train_dataset.glove)
    model = model.to(device)
    model.load_state_dict(torch.load(model_file_name))
    lines = list()
    i = 0
    n = len(sentences)
    for sentence in sentences:
        i += 1
        print("Sentence {} / {}".format(i, n))
        tokens_vector = train_dataset.vectorize_tokens([w[0] for w in sentence])
        pos_vector = train_dataset.vectorize_pos([w[1] for w in sentence])
        scores = model(tokens_vector, pos_vector)
        mst, _ = decode_mst(scores.detach().numpy(), scores.shape[0], has_labels=False)

        for i in range(1,len(mst)):
            line = [str(i), sentence[i][0], "_", sentence[i][1], "_", "_", str(mst[i]), "_", "_", "_\n"]
            lines.append("\t".join(line))
        lines.append("\n")

    with open(comp_output_path , "w") as output:
        output.writelines(line)