"""Shared MIMIC-IV-MM data, model and checkpoint interfaces."""
import json
import random
from pathlib import Path


def set_seed(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(name):
    import torch
    if name == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if name.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; use --device cpu.')
    return name


def load_dataset(split, config):
    from dataset.MIMIC_IV_MM_npy_dataset import NPY_Large_MultimodalPred_Dataset
    return NPY_Large_MultimodalPred_Dataset(
        split, config['task'], config['large_image_model_name'],
        config['large_text_model_name'], config['large_time_model_name'],
        large_data_path=config.get('large_data_path', './data/large_embed'),
        small_data_path=config.get('small_data_path', './data/MultimodalPred_input'),
        feature_dims={'image': config['image_dim'], 'text': config['text_dim'],
                      'time': config['time_dim'], 'small': 1536})


def build_model(config, device, small_model_path=None):
    from models.PPO_MultiModal import PPOMultiModalModel
    return PPOMultiModalModel(
        img_dim=config['image_dim'], text_dim=config['text_dim'],
        time_dim=config['time_dim'], hidden_dim=config['hidden_dim'],
        num_modality=3, num_classes=2, cls_dropout=config['cls_dropout'],
        device=device, gate_temperature=config['gate_temperature'],
        small_model_path=small_model_path).to(device)


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
