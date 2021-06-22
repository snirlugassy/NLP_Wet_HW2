import torch
import torchtext
from torch.utils.data.dataset import Dataset
from collections import Counter, OrderedDict
from pos_data_reader import PosDataReader


# These are not relevant for our POS tagger but might be usefull for HW2
UNKNOWN_TOKEN = "<unk>"
# Optional: this is used to pad a batch of sentences in different lengths.
PAD_TOKEN = "<pad>"
# ROOT_TOKEN = PAD_TOKEN # this can be used if you are not padding your batches
# ROOT_TOKEN = "<root>" # use this if you are padding your batches and want a special token for ROOT
SPECIAL_TOKENS = [PAD_TOKEN, UNKNOWN_TOKEN]


class PosDataset(Dataset):
    def __init__(self, word_counts, pos_counts, file: str,
                 padding=False, word_embeddings=None):
        super().__init__()
        # self.subset = subset # One of the following: [train, test]
        self.data_reader = PosDataReader(file)
        self.vocab_size = len(word_counts)
        self.word_counts = word_counts
        self.pos_counts = pos_counts
        if word_embeddings:
            self.word_idx_mappings, self.idx_word_mappings, self.word_vectors = word_embeddings()
        else:
            self.word_idx_mappings, self.idx_word_mappings, self.word_vectors = self.init_word_embeddings()

        self.pos_idx_mappings, self.idx_pos_mappings = self.init_pos_vocab(pos_counts)
        self.pad_idx = self.word_idx_mappings.get(PAD_TOKEN)
        self.unknown_idx = self.word_idx_mappings.get(UNKNOWN_TOKEN)
        self.word_vector_dim = self.word_vectors.size(-1)
        self.sentence_lens = [len(sentence)
                              for sentence in self.data_reader.sentences]
        self.max_seq_len = max(self.sentence_lens)
        self.sentences_dataset = self.convert_sentences_to_dataset(padding)
        self._length = len(self.sentences_dataset)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        word_embed_idx, pos_embed_idx, sentence_len = self.sentences_dataset[index]
        return word_embed_idx, pos_embed_idx, sentence_len

    def init_word_embeddings(self):
        # glove = Vocab(Counter(word_counts), vectors="glove.6B.300d", specials=SPECIAL_TOKENS)
        # return glove.stoi, glove.itos, glove.vectors
        words_sorted_by_count = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        ordered_word_counts_dict = OrderedDict(words_sorted_by_count)
        vocab = torchtext.vocab.vocab(ordered_word_counts_dict, min_freq=1)
        return vocab.stoi, vocab.itos, vocab.vectors

    def get_word_embeddings(self):
        return self.word_idx_mappings, self.idx_word_mappings, self.word_vectors

    def init_pos_vocab(self, pos_counts):
        idx_pos_mappings = sorted(
            [self.word_idx_mappings.get(token) for token in SPECIAL_TOKENS])
        pos_idx_mappings = {
            self.idx_word_mappings[idx]: idx for idx in idx_pos_mappings}

        for i, pos in enumerate(sorted(pos_counts.keys())):
            # pos_idx_mappings[str(pos)] = int(i)
            pos_idx_mappings[str(pos)] = int(i+len(SPECIAL_TOKENS))
            idx_pos_mappings.append(str(pos))
        print("idx_pos_mappings -", idx_pos_mappings)
        print("pos_idx_mappings -", pos_idx_mappings)
        return pos_idx_mappings, idx_pos_mappings

    def get_pos_vocab(self):
        return self.pos_idx_mappings, self.idx_pos_mappings

    def convert_sentences_to_dataset(self, padding):
        sentence_word_idx_list = list()
        sentence_pos_idx_list = list()
        sentence_len_list = list()
        for sentence_idx, sentence in enumerate(self.data_reader.sentences):
            words_idx_list = []
            pos_idx_list = []
            for word, pos in sentence:
                words_idx_list.append(self.word_idx_mappings.get(word))
                pos_idx_list.append(self.pos_idx_mappings.get(pos))
            sentence_len = len(words_idx_list)
            # if padding:
            #     while len(words_idx_list) < self.max_seq_len:
            #         words_idx_list.append(self.word_idx_mappings.get(PAD_TOKEN))
            #         pos_idx_list.append(self.pos_idx_mappings.get(PAD_TOKEN))
            sentence_word_idx_list.append(torch.tensor(
                words_idx_list, dtype=torch.long, requires_grad=False))
            sentence_pos_idx_list.append(torch.tensor(
                pos_idx_list, dtype=torch.long, requires_grad=False))
            sentence_len_list.append(sentence_len)

        # if padding:
        #     all_sentence_word_idx = torch.tensor(sentence_word_idx_list, dtype=torch.long)
        #     all_sentence_pos_idx = torch.tensor(sentence_pos_idx_list, dtype=torch.long)
        #     all_sentence_len = torch.tensor(sentence_len_list, dtype=torch.long, requires_grad=False)
        #     return TensorDataset(all_sentence_word_idx, all_sentence_pos_idx, all_sentence_len)

        return {i: sample_tuple for i, sample_tuple in enumerate(zip(sentence_word_idx_list,
                                                                     sentence_pos_idx_list,
                                                                     sentence_len_list))}
