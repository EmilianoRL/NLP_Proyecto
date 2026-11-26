"""
Persistencia en Google Drive (recomendación operativa del profesor).

`save_experiment` y `load_experiment` conservan la firma y el formato del código
entregado por el profesor. Se agregan, sin romper compatibilidad:
  - 'epoch' y 'extra' opcionales dentro del checkpoint (para reanudar),
  - una copia legible del historial en JSON,
  - `resume_experiment` para continuar un entrenamiento que se cortó.

Convención de archivos por modelo (p. ej. model_name='modeloA'):
  modeloA.pt            -> MEJOR época según val_loss  (se usa para evaluar en test)
  modeloA_last.pt       -> ÚLTIMA época terminada       (se usa para reanudar)
  modeloA_history.json  -> train_loss / val_loss legibles
"""
import os
import json
import torch

SAVE_DIR = '/content/drive/MyDrive/Proyecto_NLP_NMT'


def setup_save_dir():
    """
    1. Montar Google Drive y crear carpeta del proyecto (bloque del profesor).
    Fuera de Colab usa ./outputs para poder probar el código localmente.
    """
    global SAVE_DIR
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except ImportError:
        SAVE_DIR = os.path.abspath('outputs')
        print('No estamos en Colab: se guardará en', SAVE_DIR)
    os.makedirs(SAVE_DIR, exist_ok=True)
    return SAVE_DIR


def _to_float_lists(history):
    # np.mean regresa np.float64; torch.load(weights_only=True) no acepta tipos de numpy
    return {k: [float(v) for v in vals] for k, vals in history.items()}


def save_experiment(model, optimizer, history, model_name, epoch=None, extra=None):
    """2. Guardar checkpoint y curvas de pérdida (formato del profesor + epoch/extra)."""
    history = _to_float_lists(history)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,  # dict con {'train_loss': [...], 'val_loss': [...]}
        'epoch': epoch,
        'extra': extra or {},
    }
    torch.save(checkpoint, os.path.join(SAVE_DIR, f"{model_name}.pt"))
    base_name = model_name.replace('_last', '')
    with open(os.path.join(SAVE_DIR, f"{base_name}_history.json"), 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Checkpoint guardado en Drive: {model_name}.pt")


def load_experiment(model, model_name, device='cuda'):
    """3. Recargar el modelo en una sesión nueva para evaluar BLEU o graficar (código del profesor)."""
    path = os.path.join(SAVE_DIR, f"{model_name}.pt")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, checkpoint['history']


def resume_experiment(model, optimizer, model_name, device='cuda'):
    """
    Si existe `<model_name>_last.pt`, recarga pesos, optimizador e historial para
    continuar desde la siguiente época. Si no, empieza de cero.

    Regresa: start_epoch, history, extra
    """
    path = os.path.join(SAVE_DIR, f"{model_name}_last.pt")
    if not os.path.exists(path):
        print(f'Sin checkpoint previo de {model_name}: se entrena desde la época 1.')
        return 0, {'train_loss': [], 'val_loss': []}, {}
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f'Reanudando {model_name} desde la época {start_epoch + 1} '
          f'(última val_loss = {checkpoint["history"]["val_loss"][-1]:.4f})')
    return start_epoch, checkpoint['history'], checkpoint.get('extra', {})
