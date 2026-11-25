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

```bash
git clone <https://github.com/EmilianoRL/NLP_Proyecto.git>
cd <NLP_Proyecto>
pip install -r requirements.txt
jupyter notebook
```

Cada notebook descarga el dataset y genera la partición con el bloque de la primera celda. Se recomienda Google Colab con GPU.

## Resultados

_TODO_
