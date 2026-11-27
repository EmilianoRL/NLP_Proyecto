# Evaluación Comparativa de Arquitecturas Seq2Seq en Traducción Automática Neuronal (EN → ES)

Proyecto Final Integrador — Procesamiento de Lenguaje Natural (TC3009C), Tecnológico de Monterrey.

## Integrantes

- Diego Octavio Arias Incháustegui (A00838285)
- Emiliano Ruiz López (A01659693)

## Descripción

Comparación de cuatro arquitecturas para traducción EN → ES sobre el corpus Tatoeba/ManyThings (`spa.txt`):

- **Modelo A:** BiLSTM Encoder–Decoder con atención (desde cero)
- **Modelo B:** Transformer compacto (desde cero)
- **Modelo C:** Transformer base (desde cero)
- **Modelo D:** Fine-Tuning de un modelo preentrenado

## Auditoría de datos

Partición determinista 80/10/10 con `random.Random(23)`.

| Partición | Muestras |
|-----------|----------|
| Train | _TODO_ |
| Validation | _TODO_ |
| Test | _TODO_ |

**SHA-256 Test Audit ID:** `TODO`

## Reproducción

### En Google Colab (recomendado)

1. Abrir cada notebook de `notebooks/` en Colab (pestaña *GitHub* → `EmilianoRL/NLP_Proyecto`), en orden.
2. Ejecutar todo. La primera celda clona este repositorio y monta Google Drive.
3. Todo lo que se reutiliza entre notebooks (copia del corpus, vocabularios, checkpoints e historiales de pérdida,
   figuras) se guarda en `MyDrive/Proyecto_NLP_NMT/`. Si la sesión se desconecta, los notebooks retoman
   desde el último checkpoint sin reentrenar.

### Local

```bash
git clone https://github.com/EmilianoRL/NLP_Proyecto.git
cd NLP_Proyecto
pip install -r requirements.txt
jupyter notebook
```

Fuera de Colab, las salidas se guardan en `./outputs/`.

## Estructura del repositorio

```
common/
  data.py         descarga del corpus y protocolo obligatorio de filtrado y partición 80/10/10
  preprocess.py   clean(): minúsculas, sin acentos, sin puntuación (también se usa para normalizar BLEU)
  vocab.py        build_vocab(): eng_word2idx / spa_word2idx construidos solo con train
  datasets.py     MyDataset y collate_fn (truncado a MAX_SEQ_LEN=128, padding con <pad>=0)
  checkpoint.py   guardado / recarga / reanudación de checkpoints en Google Drive
  metrics.py      BLEU global y por longitud (sacrebleu), perplejidad, latencia
  model_a.py      Modelo A: encoder BiLSTM + decoder LSTM con atención de producto punto
notebooks/
  01_datos_auditoria.ipynb   partición, Test Audit ID, exploración y vocabularios
  02_modeloA.ipynb           entrenamiento y evaluación del Modelo A
  05_modeloD.ipynb           fine-tuning de Helsinki-NLP/opus-mt-en-es (Modelo D), dos configuraciones
proyecto_final.ipynb         setup inicial (protocolo de datos)
requirements.txt
```

## Resultados

_TODO_
