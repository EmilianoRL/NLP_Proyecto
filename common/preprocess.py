import re
import unicodedata


def clean(text):
    # minúsculas, sin acentos y sin puntuación
    # se usa para los modelos A, B y C, y para normalizar predicciones y referencias en el BLEU
    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')   # á -> a, ñ -> n
    text = re.sub(r"['’`]", '', text)                                    # don't -> dont
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return ' '.join(text.split())


def tokenize(text):
    return clean(text).split()
