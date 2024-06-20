from cog import BasePredictor, Input, Path
from typing import Dict
from pathlib import Path
import tempfile
import torch
import torchaudio
import librosa
import subprocess
import os
import soundfile as sf
from pyupload.main import CatboxUploader
from scipy.io.wavfile import write as write_wav

SAMPLE_RATE = 16000

# TODO: Replicate is unable to parse & render if we return plain Audio, hence used this method of temporary output url, see if it can resolve        
class Predictor(BasePredictor):
    def setup(self):
        self.mars5, self.config_class = torch.hub.load('Camb-ai/mars5-tts', 'mars5_english', trust_repo=True)
        print(">>>>> Model Loaded")

    def predict(
        self,
        text: str = Input(description="Text to synthesize"),
        ref_audio_file: Path = Input(description='Reference audio file to clone from <= 10 seconds', default="https://files.catbox.moe/be6df3.wav"),
        ref_audio_transcript: str = Input(description='Text in the reference audio file', default="We actually haven't managed to meet demand."),
    ) -> str:
        
        print(f">>>> Ref Audio file: {ref_audio_file}; ref_transcript: {ref_audio_transcript}")
        
        # Load the reference audio
        wav, sr = librosa.load(ref_audio_file, sr=self.mars5.sr, mono=True)
        wav = torch.from_numpy(wav)

        # configuration for the TTS model
        deep_clone = True
        cfg = self.config_class(deep_clone=deep_clone, rep_penalty_window=100, top_k=100, temperature=0.7, freq_penalty=3)

        # Generate the synthesized audio
        print(f">>> Running inference")
        ar_codes, wav_out = self.mars5.tts(text, wav, ref_audio_transcript, cfg=cfg)
        print(f">>>>> Done with inference")
        
        output_path = "/tmp/aud.mp3"
        write_wav(output_path, self.mars5.sr, wav_out.numpy())
        
        output_file_url = CatboxUploader(output_path).execute()
        print(f">>>> Output file url: {output_file_url}")
        return output_file_url
