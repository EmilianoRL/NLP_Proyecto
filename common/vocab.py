"""
Vocabularios de palabras (eng_word2idx, spa_word2idx) construidos SOLO con train_pairs.

Sigue la "Guía de Implementación" de las slides Transformer Parte 2 (build_vocab):
  - envolver cada oración con '<sos> ... <eos>' antes de construir el vocabulario,
  - tokenizar con .split(), contar con collections.Counter y ordenar de mayor a menor frecuencia,
  - '<pad>': 0  -> coincide con pad_idx=0 en máscaras y padding_value=0 en pad_sequence,
  - '<unk>': 1  -> palabras desconocidas en val/test,
  - resto de tokens (incluidos <sos> y <eos>): índices consecutivos desde 2.

Como <sos> y <eos> NO tienen índice fijo, el código siempre los consulta en el
diccionario: spa_word2idx['<sos>'], spa_word2idx['<eos>'].

Agregamos un umbral mínimo de frecuencia (`count_threshold`), como el vocabulario
cerrado del notebook de N-gramas: las palabras con menos apariciones en train se tratan como <unk>.
"""
import json
from collections import Counter

PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN = '<pad>', '<unk>', '<sos>', '<eos>'
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]
PAD_IDX, UNK_IDX = 0, 1


def wrap(sentence):
    """'el gato negro' -> ['<sos>', 'el', 'gato', 'negro', '<eos>']"""
    return f'{SOS_TOKEN} {sentence} {EOS_TOKEN}'.split()


def build_vocab(sentences, count_threshold=1):
    """
    `sentences`: oraciones de TRAIN ya limpias (texto).
    Regresa el diccionario palabra -> índice.
    """
    counter = Counter()
    for sentence in sentences:
        counter.update(wrap(sentence))
    word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, count in counter.most_common():          # de mayor a menor frecuencia
        if count >= count_threshold or word in (SOS_TOKEN, EOS_TOKEN):
            word2idx[word] = len(word2idx)
    return word2idx


def invert(word2idx):
    """índice -> palabra"""
    return {i: w for w, i in word2idx.items()}


def sentence_to_ids(sentence, word2idx):
    """Oración limpia -> [<sos>, ..., <eos>] en índices. Lo que no esté en train se vuelve <unk>."""
    return [word2idx.get(tok, UNK_IDX) for tok in wrap(sentence)]


def ids_to_sentence(ids, idx2word):
    """Índices -> texto. Se detiene en <eos> y omite <pad>/<sos>."""
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
    """Porcentaje de palabras (sin contar <sos>/<eos>) que se convertirían en <unk>."""
    tokens = [tok for s in sentences for tok in s.split()]
    unk = sum(1 for tok in tokens if tok not in word2idx)
    return 100.0 * unk / max(len(tokens), 1)


def save_vocab(path, eng_word2idx, spa_word2idx, count_threshold):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'count_threshold': count_threshold,
                   'eng_word2idx': eng_word2idx,
                   'spa_word2idx': spa_word2idx}, f, ensure_ascii=False)
    print(f'Vocabularios guardados en {path}')


def load_vocab(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['eng_word2idx'], data['spa_word2idx'], data['count_threshold']
