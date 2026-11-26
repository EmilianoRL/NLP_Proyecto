"""
Descarga del corpus Tatoeba/ManyThings (spa.txt) y protocolo OBLIGATORIO
de filtrado y partición 80/10/10 (Sección 3 del proyecto).

La función `load_splits` reproduce línea por línea el bloque del profesor;
NO modificar su lógica: el SHA-256 Test Audit ID depende de ella.
"""
import os
import json
import random
import hashlib
import shutil
import zipfile
import urllib.request

CORPUS_URL = 'https://www.manythings.org/anki/spa-eng.zip'
CORPUS_ZIP = 'spa-eng.zip'
CORPUS_TXT = 'spa.txt'
SPLITS_JSON = 'splits.json'

# Partición oficial del equipo (obtenida por ambos integrantes con el bloque del protocolo)
EXPECTED_COUNTS = (29130, 3641, 3642)
EXPECTED_TEST_AUDIT_ID = 'dbac04ac5648'


def download_corpus(data_dir='.', cache_dir=None):
    """
    Deja `spa.txt` dentro de `data_dir`.

    Orden de búsqueda:
      1. Si `spa.txt` ya existe en `data_dir`, no hace nada.
      2. Si existe una copia del zip en `cache_dir` (Google Drive), la usa.
      3. Si no, descarga el zip de manythings.org y guarda una copia en `cache_dir`.

    Guardar el zip en Drive congela la versión del corpus: así ambos integrantes
    (y cualquier sesión futura) obtienen exactamente el mismo Test Audit ID.
    """
    txt_path = os.path.join(data_dir, CORPUS_TXT)
    zip_path = os.path.join(data_dir, CORPUS_ZIP)
    if os.path.exists(txt_path):
        print(f'Corpus ya disponible: {txt_path}')
        return txt_path

    cached_zip = os.path.join(cache_dir, CORPUS_ZIP) if cache_dir else None
    if cached_zip and os.path.exists(cached_zip):
        print(f'Usando copia del corpus guardada en Drive: {cached_zip}')
        shutil.copy(cached_zip, zip_path)
    else:
        print(f'Descargando {CORPUS_URL} ...')
        # manythings.org puede rechazar peticiones sin user-agent de navegador
        request = urllib.request.Request(CORPUS_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(request) as response, open(zip_path, 'wb') as f:
            shutil.copyfileobj(response, f)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            shutil.copy(zip_path, cached_zip)
            print(f'Copia del corpus guardada en Drive: {cached_zip}')

    with zipfile.ZipFile(zip_path) as z:
        z.extract(CORPUS_TXT, data_dir)
    print(f'Corpus listo: {txt_path}')
    return txt_path


def file_sha256(path):
    """Huella del archivo spa.txt completo (identifica la versión del corpus)."""
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def load_splits(path=CORPUS_TXT, verbose=True):
    """
    Protocolo obligatorio de carga, filtrado y partición reproducible
    (copiado del enunciado del proyecto, Listing 1).

    Regresa: train_pairs, val_pairs, test_pairs, test_hash
    Cada par es una tupla (en, es) con el texto ORIGINAL (sin limpiar).
    """
    with open(path, 'r', encoding='utf-8') as f:
        raw_pairs = [line.strip().split('\t')[:2] for line in f if '\t' in line]

    # 2. Filtrado de complejidad mínima (>= 8 palabras en inglés, <= 120 tokens)
    filtered_pairs = [
        (en, es) for en, es in raw_pairs
        if 8 <= len(en.split()) <= 120 and len(es.split()) <= 120
    ]

    # 3. Mezcla determinista con semilla fija (obligatoria para auditoría)
    rng = random.Random(23)
    rng.shuffle(filtered_pairs)

    # 4. Partición estricta 80% Train / 10% Validation / 10% Test
    n = len(filtered_pairs)
    n_train = int(0.80 * n)
    n_val = int(0.10 * n)

    train_pairs = filtered_pairs[:n_train]
    val_pairs = filtered_pairs[n_train:n_train + n_val]
    test_pairs = filtered_pairs[n_train + n_val:]

    # 5. Huella de auditoría (debe reportarse en el notebook y en el PDF)
    test_hash = hashlib.sha256("\n".join([p[0] for p in test_pairs]).encode()).hexdigest()[:12]
    if verbose:
        print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)} | Test: {len(test_pairs)}")
        print(f"SHA-256 Test Audit ID: {test_hash}")
    return train_pairs, val_pairs, test_pairs, test_hash


def save_splits_json(path, train_pairs, val_pairs, test_pairs):
    """Respaldo de la partición en JSON (mismo formato que splits.json del equipo)."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'train': train_pairs, 'val': val_pairs, 'test': test_pairs}, f, ensure_ascii=False)
    print(f'Partición guardada en {path}')


def load_splits_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        splits = json.load(f)
    train_pairs = [tuple(p) for p in splits['train']]
    val_pairs = [tuple(p) for p in splits['val']]
    test_pairs = [tuple(p) for p in splits['test']]
    test_hash = hashlib.sha256("\n".join([p[0] for p in test_pairs]).encode()).hexdigest()[:12]
    return train_pairs, val_pairs, test_pairs, test_hash


def get_splits(data_dir):
    """
    Punto de entrada para los notebooks 02-06.
      1. Si existe `splits.json` en `data_dir` (Drive), lo usa: es la partición oficial
         del equipo y no depende de que manythings.org siga ofreciendo el mismo archivo.
      2. Si no, aplica el protocolo sobre spa.txt (descargándolo o usando la copia de Drive).
    En ambos casos verifica que los tamaños y el Test Audit ID sean los oficiales.
    """
    json_path = os.path.join(data_dir, SPLITS_JSON)
    if os.path.exists(json_path):
        train_pairs, val_pairs, test_pairs, test_hash = load_splits_json(json_path)
        source = json_path
    else:
        txt_path = download_corpus('.', cache_dir=data_dir)
        train_pairs, val_pairs, test_pairs, test_hash = load_splits(txt_path, verbose=False)
        source = txt_path
    counts = (len(train_pairs), len(val_pairs), len(test_pairs))
    assert counts == EXPECTED_COUNTS and test_hash == EXPECTED_TEST_AUDIT_ID, (
        f'La partición no coincide con la oficial: {counts} {test_hash} '
        f'(esperado {EXPECTED_COUNTS} {EXPECTED_TEST_AUDIT_ID})')
    print(f'Partición cargada de {source}')
    print(f'Train: {counts[0]} | Val: {counts[1]} | Test: {counts[2]} | SHA-256 Test Audit ID: {test_hash}')
    return train_pairs, val_pairs, test_pairs


def length_bucket(en_sentence):
    """
    Grupo de longitud usado en la Tabla 1, medido en palabras de la oración
    FUENTE en inglés (mismo criterio que el filtro del protocolo: en.split()).
      'media' : 8-18 palabras
      'larga' : > 18 palabras
    """
    n_words = len(en_sentence.split())
    return 'media' if n_words <= 18 else 'larga'
