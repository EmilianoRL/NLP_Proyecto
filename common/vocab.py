# Vocabularios eng_word2idx / spa_word2idx (solo con train)
# Como en las slides de Transformer parte 2: <sos> ... <eos>, Counter, de mayor a menor frecuencia,
# <pad> = 0, <unk> = 1 y el resto desde 2. Además un mínimo de frecuencia, como en el notebook de N-gramas.
import json
from collections import Counter

PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN = '<pad>', '<unk>', '<sos>', '<eos>'
PAD_IDX, UNK_IDX = 0, 1


def wrap(sentence):
    return f'{SOS_TOKEN} {sentence} {EOS_TOKEN}'.split()


def build_vocab(sentences, count_threshold=1):
    counter = Counter()
    for sentence in sentences:
        counter.update(wrap(sentence))
    word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, count in counter.most_common():
        if count >= count_threshold or word in (SOS_TOKEN, EOS_TOKEN):
            word2idx[word] = len(word2idx)
    return word2idx


def invert(word2idx):
    return {i: w for w, i in word2idx.items()}


def sentence_to_ids(sentence, word2idx):
    return [word2idx.get(tok, UNK_IDX) for tok in wrap(sentence)]


def ids_to_sentence(ids, idx2word):
    # se detiene en <eos> y se salta <pad> y <sos>
    words = []
    for i in ids:
        word = idx2word.get(int(i), UNK_TOKEN)
        if word == EOS_TOKEN:
            break
        if word in (PAD_TOKEN, SOS_TOKEN):
            continue
        words.append(word)
    return ' '.join(words)


def oov_rate(sentences, word2idx):
    # % de palabras que quedarían como <unk>
    tokens = [tok for s in sentences for tok in s.split()]
    unk = sum(1 for tok in tokens if tok not in word2idx)
    return 100.0 * unk / max(len(tokens), 1)


def save_vocab(path, eng_word2idx, spa_word2idx, count_threshold):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'count_threshold': count_threshold,
                   'eng_word2idx': eng_word2idx,
                   'spa_word2idx': spa_word2idx}, f, ensure_ascii=False)


def load_vocab(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['eng_word2idx'], data['spa_word2idx'], data['count_threshold']
