dependencies = ['torch', 'torchaudio', 'numpy', 'vocos', 'safetensors']

import logging
import os
from pathlib import Path
from safetensors import safe_open
import torch
from inference import Mars5TTS, InferenceConfig


# Centralized checkpoint URLs for easy management and updates
CHECKPOINT_URLS = {
    "ar": "https://github.com/Camb-ai/MARS5-TTS/releases/download/v0.4/mars5_en_checkpoints_ar-3000000.pt",
    "nar": "https://github.com/Camb-ai/MARS5-TTS/releases/download/v0.3/mars5_en_checkpoints_nar-1980000.pt",
    "ar_sf": "https://github.com/Camb-ai/MARS5-TTS/releases/download/v0.4/mars5_en_checkpoints_ar-3000000.safetensors",
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

    
# Load Mars5 English model on `device`, optionally showing progress.
# This function also handles user-provided path for model checkpoints,
# supporting both .pt and .safetensors formats.

def mars5_english(pretrained=True, progress=True, device=None, ckpt_format='safetensors', ar_path=None, nar_path=None):
    
   
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")

    if not pretrained:
        raise ValueError('Only pretrained models are currently supported.')

    # Determine the format of the checkpoint based on the file extension if paths are provided
    if ar_path is not None:
        if ar_path.endswith('.pt'):
            ar_ckpt = load_checkpoint(None, progress, 'pt', ar_path)
        elif ar_path.endswith('.safetensors'):
            ar_ckpt = load_checkpoint(None, progress, 'safetensors', ar_path)
        else:
            raise NotImplementedError("Unsupported file format for ar_path. Please provide a .pt or .safetensors file.")
    else:
        ar_ckpt = load_checkpoint(CHECKPOINT_URLS[f'ar_{ckpt_format}'], progress, ckpt_format)

    if nar_path is not None:
        if nar_path.endswith('.pt'):
            nar_ckpt = load_checkpoint(None, progress, 'pt', nar_path)
        elif nar_path.endswith('.safetensors'):
            nar_ckpt = load_checkpoint(None, progress, 'safetensors', nar_path)
        else:
            raise NotImplementedError("Unsupported file format for nar_path. Please provide a .pt or .safetensors file.")
    else:
        nar_ckpt = load_checkpoint(CHECKPOINT_URLS[f'nar_{ckpt_format}'], progress, ckpt_format)

    logging.info("Initializing models...")
    return Mars5TTS(ar_ckpt, nar_ckpt, device=device), InferenceConfig

