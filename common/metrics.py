# BLEU y perplejidad
# BLEU con evaluate.load('sacrebleu') como en los notebooks de traducción de clase.
# Se aplica clean() a predicciones y referencias; las referencias son el texto original de test.
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
    preds = [clean(p) for p in predictions]
    refs = [[clean(r)] for r in references]
    try:
        return _get_metric().compute(predictions=preds, references=refs)['score']
    except Exception:
        # si no se puede descargar la métrica de Hugging Face, sacrebleu directo da lo mismo
        import sacrebleu
        return sacrebleu.corpus_bleu(preds, [[r[0] for r in refs]]).score


def bleu_report(sources, predictions, references):
    # BLEU global, en oraciones medias (8-18 palabras) y largas (>18)
    report = {'global': corpus_bleu(predictions, references), 'n_global': len(sources)}
    for bucket in ('media', 'larga'):
        idx = [i for i, s in enumerate(sources) if length_bucket(s) == bucket]
        report[bucket] = corpus_bleu([predictions[i] for i in idx], [references[i] for i in idx])
        report[f'n_{bucket}'] = len(idx)
    return report


def perplexity(total_loss, total_tokens):
    # exp de la pérdida promedio por token (total_loss es la suma, no el promedio por batch)
    return math.exp(total_loss / total_tokens)


def measure_latency(translate_one, sentences, device, warmup=10):
    # ms por oración, traduciendo de una en una (batch = 1, greedy)
    import time
    import torch
    for s in sentences[:warmup]:
        translate_one(s)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    for s in sentences:
        translate_one(s)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - start) / len(sentences)


def gpu_name():
    import torch
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'


def save_results(save_dir, model_name, results):
    # results/<modelo>.json con métricas y predicciones de test (lo usa el notebook final)
    import os
    import json
    os.makedirs(os.path.join(save_dir, 'results'), exist_ok=True)
    path = os.path.join(save_dir, 'results', f'{model_name}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print('Resultados guardados en', path)
