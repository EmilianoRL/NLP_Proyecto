# Dataset y collate_fn para los modelos A, B y C
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from .preprocess import clean
from .vocab import sentence_to_ids, PAD_IDX

MAX_SEQ_LEN = 128


def pairs_to_ids(pairs, eng_word2idx, spa_word2idx):
    # (en, es) -> ([<sos> ... <eos>], [<sos> ... <eos>]) en índices
    src_ids = [sentence_to_ids(clean(en), eng_word2idx) for en, es in pairs]
    tgt_ids = [sentence_to_ids(clean(es), spa_word2idx) for en, es in pairs]
    return src_ids, tgt_ids


class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = torch.tensor(self.x[index])
        y = torch.tensor(self.y[index])
        return x, y


def collate_fn(batch):
    xs, ys = zip(*batch)
    # cortar a MAX_SEQ_LEN
    xs = [x[:MAX_SEQ_LEN] for x in xs]
    ys = [y[:MAX_SEQ_LEN] for y in ys]
    # longitudes para pack_padded_sequence
    lengths = [len(x) for x in xs]
    eng_batch = pad_sequence(xs, batch_first=True, padding_value=PAD_IDX)
    spa_batch = pad_sequence(ys, batch_first=True, padding_value=PAD_IDX)
    return eng_batch, spa_batch, lengths
