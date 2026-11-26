# Guardado de checkpoints en Drive (basado en el código que nos dio el profe)
# modeloX.pt          -> mejor época según val_loss (para evaluar)
# modeloX_last.pt     -> última época (para continuar si se corta la sesión)
# modeloX_history.json
import os
import json
import torch

SAVE_DIR = '/content/drive/MyDrive/Proyecto_NLP_NMT'


def setup_save_dir():
    # monta Drive; fuera de Colab guarda en ./outputs
    global SAVE_DIR
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except ImportError:
        SAVE_DIR = os.path.abspath('outputs')
    os.makedirs(SAVE_DIR, exist_ok=True)
    return SAVE_DIR


def save_experiment(model, optimizer, history, model_name, epoch=None, extra=None):
    # float() porque torch.load no acepta los np.float64 que regresa np.mean
    history = {k: [float(v) for v in vals] for k, vals in history.items()}
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
    path = os.path.join(SAVE_DIR, f"{model_name}.pt")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, checkpoint['history']


def resume_experiment(model, optimizer, model_name, device='cuda'):
    # regresa (start_epoch, history, extra); si no hay checkpoint empieza de cero
    path = os.path.join(SAVE_DIR, f"{model_name}_last.pt")
    if not os.path.exists(path):
        return 0, {'train_loss': [], 'val_loss': []}, {}
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f'Continuando {model_name} desde la época {start_epoch + 1}')
    return start_epoch, checkpoint['history'], checkpoint.get('extra', {})
