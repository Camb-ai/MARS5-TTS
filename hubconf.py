dependencies = ['torch', 'torchaudio', 'numpy', 'vocos']

import logging
from pathlib import Path

import torch
from inference import Mars5TTS, InferenceConfig

ar_url = "https://github.com/Camb-ai/mars5-tts/releases/download/v0.1-checkpoints/mars5_en_checkpoints_ar-1680000.pt"
nar_url = "https://github.com/Camb-ai/mars5-tts/releases/download/v0.1-checkpoints/mars5_en_checkpoints_nar-1260000.pt"

def mars5_english(pretrained=True, progress=True, device=None, ar_path=None, nar_path=None) -> Mars5TTS:
    """ Load mars5 english model on `device`, optionally show `progress`. """
    if device is None: device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")
    if pretrained == False: raise AssertionError('Only pretrained model currently supported.')
    logging.info("Loading AR checkpoint...")
    if ar_path is None:
        ar_ckpt = torch.hub.load_state_dict_from_url(
            ar_url, progress=progress, check_hash=False, map_location='cpu'
        )
    else: ar_ckpt = torch.load(str(ar_path), map_location='cpu')

    logging.info("Loading NAR checkpoint...")
    if nar_path is None:
        nar_ckpt = torch.hub.load_state_dict_from_url(
            nar_url, progress=progress, check_hash=False, map_location='cpu'
        )
    else: nar_ckpt = torch.load(str(nar_path), map_location='cpu')
    logging.info("Initializing modules...")
    mars5 = Mars5TTS(ar_ckpt, nar_ckpt, device=device)
    return mars5, InferenceConfig

