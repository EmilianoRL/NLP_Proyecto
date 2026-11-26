# Descarga del corpus y partición 80/10/10 (protocolo del enunciado)
import os
import json
import random
import hashlib
import shutil
import subprocess
import zipfile
import urllib.request

CORPUS_URL = 'https://www.manythings.org/anki/spa-eng.zip'
CORPUS_ZIP = 'spa-eng.zip'
CORPUS_TXT = 'spa.txt'
SPLITS_JSON = 'splits.json'

# partición que obtuvimos los dos con el bloque del enunciado
EXPECTED_COUNTS = (29130, 3641, 3642)
EXPECTED_TEST_AUDIT_ID = 'dbac04ac5648'


def download_corpus(data_dir='.', cache_dir=None):
    # usa el zip local; si no, la copia en Drive; si no, lo descarga con wget como en el enunciado
    txt_path = os.path.join(data_dir, CORPUS_TXT)
    zip_path = os.path.join(data_dir, CORPUS_ZIP)
    cached_zip = os.path.join(cache_dir, CORPUS_ZIP) if cache_dir else None

    if not os.path.exists(zip_path):
        if cached_zip and os.path.exists(cached_zip):
            shutil.copy(cached_zip, zip_path)
        elif shutil.which('wget'):
            # manythings.org rechaza urllib (error 406) pero acepta wget
            subprocess.run(['wget', '-q', CORPUS_URL, '-O', zip_path], check=True)
        else:
            request = urllib.request.Request(CORPUS_URL, headers={'User-Agent': 'Wget/1.21'})
            with urllib.request.urlopen(request) as response, open(zip_path, 'wb') as f:
                shutil.copyfileobj(response, f)

    # copia en Drive para que los dos usemos siempre el mismo archivo
    if cached_zip and not os.path.exists(cached_zip):
        os.makedirs(cache_dir, exist_ok=True)
        shutil.copy(zip_path, cached_zip)

    if not os.path.exists(txt_path):
        with zipfile.ZipFile(zip_path) as z:
            z.extract(CORPUS_TXT, data_dir)
    return txt_path


def file_sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def load_splits(path=CORPUS_TXT, verbose=True):
    # mismo código que el bloque del enunciado, dentro de una función
    with open(path, 'r', encoding='utf-8') as f:
        raw_pairs = [line.strip().split('\t')[:2] for line in f if '\t' in line]

    filtered_pairs = [
        (en, es) for en, es in raw_pairs
        if 8 <= len(en.split()) <= 120 and len(es.split()) <= 120
    ]

    rng = random.Random(23)
    rng.shuffle(filtered_pairs)

    n = len(filtered_pairs)
    n_train = int(0.80 * n)
    n_val = int(0.10 * n)

    train_pairs = filtered_pairs[:n_train]
    val_pairs = filtered_pairs[n_train:n_train + n_val]
    test_pairs = filtered_pairs[n_train + n_val:]

    test_hash = hashlib.sha256("\n".join([p[0] for p in test_pairs]).encode()).hexdigest()[:12]
    if verbose:
        print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)} | Test: {len(test_pairs)}")
        print(f"SHA-256 Test Audit ID: {test_hash}")
    return train_pairs, val_pairs, test_pairs, test_hash


def save_splits_json(path, train_pairs, val_pairs, test_pairs):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'train': train_pairs, 'val': val_pairs, 'test': test_pairs}, f, ensure_ascii=False)


def load_splits_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        splits = json.load(f)
    train_pairs = [tuple(p) for p in splits['train']]
    val_pairs = [tuple(p) for p in splits['val']]
    test_pairs = [tuple(p) for p in splits['test']]
    test_hash = hashlib.sha256("\n".join([p[0] for p in test_pairs]).encode()).hexdigest()[:12]
    return train_pairs, val_pairs, test_pairs, test_hash


def get_splits(data_dir):
    # para los notebooks de modelos: usa splits.json de Drive y, si no existe, el protocolo sobre spa.txt
    json_path = os.path.join(data_dir, SPLITS_JSON)
    if os.path.exists(json_path):
        train_pairs, val_pairs, test_pairs, test_hash = load_splits_json(json_path)
    else:
        txt_path = download_corpus('.', cache_dir=data_dir)
        train_pairs, val_pairs, test_pairs, test_hash = load_splits(txt_path, verbose=False)
    counts = (len(train_pairs), len(val_pairs), len(test_pairs))
    assert counts == EXPECTED_COUNTS and test_hash == EXPECTED_TEST_AUDIT_ID, \
        f'La partición no coincide: {counts} {test_hash}'
    print(f'Train: {counts[0]} | Val: {counts[1]} | Test: {counts[2]} | SHA-256 Test Audit ID: {test_hash}')
    return train_pairs, val_pairs, test_pairs


def length_bucket(en_sentence):
    # media: 8-18 palabras en inglés, larga: más de 18
    return 'media' if len(en_sentence.split()) <= 18 else 'larga'
