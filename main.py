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
import matplotlib.pyplot as plt
from chu_liu_edmonds import decode_mst
from utils import get_vocabs

data_dir = os.getcwd()
print("* Data directory - ", data_dir)

path_train = os.path.join(data_dir, "train.labeled")
path_test = os.path.join(data_dir, "test.labeled")

word_counts, pos_counts = get_vocabs([path_train, path_test])