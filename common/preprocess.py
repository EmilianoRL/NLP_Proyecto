"""
Normalización de texto.

`clean` se usa en DOS lugares y debe ser exactamente la misma función:
  1. Preprocesar las oraciones de entrada/salida de los modelos desde cero (A, B, C).
  2. Normalizar predicciones y referencias de los 4 modelos antes de calcular BLEU
     (Regla 3: normalización simétrica).
"""
import re
import unicodedata


def clean(text):
    """
    minúsculas -> sin acentos -> sin signos de puntuación -> espacios simples.

    Ejemplo:
        >>> clean("¿Dónde está el niño? ¡No lo sé, Tom!")
        'donde esta el nino no lo se tom'
    """
    # 1. minúsculas
    text = text.lower()
    # 2. quitar acentos y diacríticos: 'á' -> 'a', 'ñ' -> 'n', 'ü' -> 'u'
    #    (NFD separa la letra de su acento; luego descartamos las marcas 'Mn')
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    # 3. apóstrofos se eliminan sin separar la palabra: "don't" -> "dont", "Tom's" -> "toms"
    text = re.sub(r"['’`]", '', text)
    # 4. cualquier otro símbolo que no sea letra o número se vuelve espacio
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # 5. colapsar espacios múltiples
    return ' '.join(text.split())


def tokenize(text):
    """Limpia y separa por espacios (lista de palabras)."""
    return clean(text).split()
