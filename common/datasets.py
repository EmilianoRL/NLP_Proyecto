"""
Dataset y collate_fn para los modelos desde cero (A, B y C).
  - MyDataset: mismo patrón que el notebook de POS tagging de la clase.
  - collate_fn: "Batching dinámico" de las slides Transformer Parte 2:
    truncar a MAX_SEQ_LEN = 128 y rellenar con pad_sequence(..., batch_first=True, padding_value=0).
"""
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from .preprocess import clean
from .vocab import sentence_to_ids, PAD_IDX

MAX_SEQ_LEN = 128


def pairs_to_ids(pairs, eng_word2idx, spa_word2idx):
    """
    (en, es) en texto original -> (ids_en, ids_es), ambos como [<sos>, ..., <eos>].
    eng_batch es la entrada del encoder; spa_batch se parte en target_input / target_output.
    """
    src_ids = [sentence_to_ids(clean(en), eng_word2idx) for en, es in pairs]
    tgt_ids = [sentence_to_ids(clean(es), spa_word2idx) for en, es in pairs]
    return src_ids, tgt_ids


class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x   # listas de ids en inglés
        self.y = y   # listas de ids en español

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = torch.tensor(self.x[index])
        y = torch.tensor(self.y[index])
        return x, y


def collate_fn(batch):
    # separar xs y ys, truncando a MAX_SEQ_LEN
    xs, ys = zip(*batch)
    xs = [x[:MAX_SEQ_LEN] for x in xs]
    ys = [y[:MAX_SEQ_LEN] for y in ys]
    # longitudes reales (para pack_padded_sequence en el Modelo A)
    lengths = [len(x) for x in xs]
    # padding dinámico por minibatch con <pad> = 0
    eng_batch = pad_sequence(xs, batch_first=True, padding_value=PAD_IDX)
    spa_batch = pad_sequence(ys, batch_first=True, padding_value=PAD_IDX)
    return eng_batch, spa_batch, lengths
