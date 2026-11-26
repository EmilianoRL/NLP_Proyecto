"""
Métricas de evaluación en test, comunes a los 4 modelos.

BLEU: se usa `evaluate.load('sacrebleu')`, igual que en los notebooks de traducción
de la clase (cap. 15). Antes de calcularlo se aplica `clean` tanto a predicciones
como a referencias (Regla 3: normalización simétrica). Las referencias son el texto
ORIGINAL de test_pairs limpiado, nunca la versión pasada por el vocabulario
(eso metería <unk> en la referencia e inflaría el BLEU).
"""
import math

from .preprocess import clean
from .data import length_bucket

_metric = None


def _get_metric():
    global _metric
    if _metric is None:
        from evaluate import load
        _metric = load('sacrebleu')
    return _metric


def corpus_bleu(predictions, references):
    """BLEU de corpus sobre textos normalizados con `clean`."""
    preds = [clean(p) for p in predictions]
    refs = [[clean(r)] for r in references]   # sacrebleu espera una lista de referencias por oración
    try:
        return _get_metric().compute(predictions=preds, references=refs)['score']
    except Exception:
        # Sin acceso al Hub de Hugging Face: misma métrica llamando directo a la librería
        # sacrebleu (evaluate.load('sacrebleu') es un envoltorio de esta función).
        import sacrebleu
        return sacrebleu.corpus_bleu(preds, [[r[0] for r in refs]]).score


def bleu_report(sources, predictions, references):
    """
    BLEU global y por longitud de la oración fuente (Tabla 1):
      media = 8-18 palabras en inglés, larga = > 18 palabras.
    `sources`, `predictions`, `references` son listas alineadas de texto original.
    """
    report = {'global': corpus_bleu(predictions, references), 'n_global': len(sources)}
    for bucket in ('media', 'larga'):
        idx = [i for i, s in enumerate(sources) if length_bucket(s) == bucket]
        report[bucket] = corpus_bleu([predictions[i] for i in idx], [references[i] for i in idx])
        report[f'n_{bucket}'] = len(idx)
    return report


def perplexity(total_loss, total_tokens):
    """
    PPL = exp(cross-entropy promedio por token).
    Es la misma perplejidad del notebook de N-gramas, (prod 1/P)^(1/N), escrita con logaritmos.
    `total_loss` debe ser la SUMA de la pérdida sobre todos los tokens (no el promedio por batch).
    """
    return math.exp(total_loss / total_tokens)
