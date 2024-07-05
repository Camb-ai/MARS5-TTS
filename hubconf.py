dependencies = ['torch', 'torchaudio', 'numpy', 'vocos', 'safetensors']

import logging
import os
from pathlib import Path
from safetensors import safe_open

import torch
from inference import Mars5TTS, InferenceConfig

ar_url = "https://github.com/Camb-ai/MARS5-TTS/releases/download/v0.4/mars5_en_checkpoints_ar-3000000.pt"
nar_url = "https://github.com/Camb-ai/MARS5-TTS/releases/download/v0.3/mars5_en_checkpoints_nar-1980000.pt"

ar_sf_url = "https://github.com/Camb-ai/MARS5-TTS/releases/download/v0.4/mars5_en_checkpoints_ar-3000000.safetensors"
nar_sf_url = "https://github.com/Camb-ai/MARS5-TTS/releases/download/v0.3/mars5_en_checkpoints_nar-1980000.safetensors"

def mars5_english(pretrained=True, progress=True, device=None, ckpt_format='safetensors',
                  ar_path=None, nar_path=None) -> Mars5TTS:
    """ Load mars5 english model on `device`, optionally show `progress`. """
    if device is None: device = 'cuda' if torch.cuda.is_available() else 'cpu'

    assert ckpt_format in ['safetensors', 'pt'], "checkpoint format must be 'safetensors' or 'pt'"
    
    logging.info(f"Using device: {device}")
    if pretrained == False: raise AssertionError('Only pretrained model currently supported.')
    logging.info("Loading AR checkpoint...")

    if ar_path is None:
        if ckpt_format == 'safetensors':
            ar_ckpt = _load_safetensors_ckpt(ar_sf_url, progress=progress)
        elif ckpt_format == 'pt':
            ar_ckpt = torch.hub.load_state_dict_from_url(
                ar_url, progress=progress, check_hash=False, map_location='cpu'
            )
    else: ar_ckpt = torch.load(str(ar_path), map_location='cpu')

    logging.info("Loading NAR checkpoint...")
    if nar_path is None:
        if ckpt_format == 'safetensors':
            nar_ckpt = _load_safetensors_ckpt(nar_sf_url, progress=progress)
        elif ckpt_format == 'pt':
            nar_ckpt = torch.hub.load_state_dict_from_url(
                nar_url, progress=progress, check_hash=False, map_location='cpu'
            )
    else: nar_ckpt = torch.load(str(nar_path), map_location='cpu')
    logging.info("Initializing modules...")
    mars5 = Mars5TTS(ar_ckpt, nar_ckpt, device=device)
    return mars5, InferenceConfig


def _load_safetensors_ckpt(url, progress):
    """ Loads checkpoint from a safetensors file """
    hub_dir = torch.hub.get_dir()
    model_dir = os.path.join(hub_dir, 'checkpoints')
    os.makedirs(model_dir, exist_ok=True)
    parts = torch.hub.urlparse(url)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        # download it
        torch.hub.download_url_to_file(url, cached_file, None, progress=progress)
    # load checkpoint
    ckpt = {}
    with safe_open(cached_file, framework='pt', device='cpu') as f:
        metadata = f.metadata()
        ckpt['vocab'] = {'texttok.model': metadata['texttok.model'], 'speechtok.model': metadata['speechtok.model']}
        ckpt['model'] = {}
        for k in f.keys(): ckpt['model'][k] = f.get_tensor(k)
    return ckpt
