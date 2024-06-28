import logging
import os
from pathlib import Path
from safetensors import safe_open
import torch
from inference import Mars5TTS, InferenceConfig

dependencies = ['torch', 'torchaudio', 'numpy', 'vocos', 'safetensors']

# Centralized checkpoint URLs for easy management and updates
CHECKPOINT_URLS = {
    "ar": "https://github.com/Camb-ai/MARS5-TTS/releases/download/v0.3/mars5_en_checkpoints_ar-2000000.pt",
    "nar": "https://github.com/Camb-ai/MARS5-TTS/releases/download/v0.3/mars5_en_checkpoints_nar-1980000.pt",
    "ar_sf": "https://github.com/Camb-ai/MARS5-TTS/releases/download/v0.3/mars5_en_checkpoints_ar-2000000.safetensors",
    "nar_sf": "https://github.com/Camb-ai/MARS5-TTS/releases/download/v0.3/mars5_en_checkpoints_nar-1980000.safetensors"
}

def load_checkpoint(url, progress=True, ckpt_format='pt'):
    """ Helper function to download and load a checkpoint, reducing duplication """
    hub_dir = torch.hub.get_dir()
    model_dir = os.path.join(hub_dir, 'checkpoints')
    os.makedirs(model_dir, exist_ok=True)
    parts = torch.hub.urlparse(url)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)

    if not os.path.exists(cached_file):
        torch.hub.download_url_to_file(url, cached_file, None, progress=progress)

    if ckpt_format == 'safetensors':
        return _load_safetensors_ckpt(cached_file)
    else:
        return torch.load(cached_file, map_location='cpu')

def _load_safetensors_ckpt(file_path):
    """ Loads a safetensors checkpoint file """
    ckpt = {}
    with safe_open(file_path, framework='pt', device='cpu') as f:
        metadata = f.metadata()
        ckpt['vocab'] = {'texttok.model': metadata['texttok.model'], 'speechtok.model': metadata['speechtok.model']}
        ckpt['model'] = {k: f.get_tensor(k) for k in f.keys()}
    return ckpt

def mars5_english(pretrained=True, progress=True, device=None, ckpt_format='safetensors', ar_path=None, nar_path=None):
    """ Load Mars5 English model on `device`, optionally show `progress`. """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")

    if not pretrained:
        raise ValueError('Only pretrained models are currently supported.')

    ar_ckpt = load_checkpoint(CHECKPOINT_URLS[f'ar_{ckpt_format}'], progress, ckpt_format) if ar_path is None else torch.load(ar_path, map_location='cpu')
    nar_ckpt = load_checkpoint(CHECKPOINT_URLS[f'nar_{ckpt_format}'], progress, ckpt_format) if nar_path is None else torch.load(nar_path, map_location='cpu')

    logging.info("Initializing models...")
    return Mars5TTS(ar_ckpt, nar_ckpt, device=device), InferenceConfig
