# Mars 5 TTS

This is the repo for the Mars 5 English TTS model from Camb.AI.
The model follows a two-stage AR-NAR pipeline similar to other large TTS models, but has a slightly different NAR component (see more info in the [docs](docs/architecture.md)). 
Mars 5 can generate speech from a snippet of text in the voice of any provided reference, where only 5 seconds of reference audio is needed.

Links:
- [Camb.AI website](https://camb.ai/) (use Mars5 to dub videos)
- Technical writeup: _to come_
- Colab quickstart: <a target="_blank" href="https://colab.research.google.com/github/Camb-ai/mars5-tts/blob/master/mars5_demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
- Demo page with samples: _to come_


![Mars 5 simplified diagram](docs/assets/simplified_diagram.png)

**Figure**: the high-level architecture flow of Mars 5. Given text and a reference audio, coarse (L0) encodec speech features are obtained through an autoregressive transformer model. Then, the text, reference, and coarse features are refined in a multinomial DDPM model to produce the remaining encodec codebook values. The output of the DDPM is then vocoded to produce the final audio.


## Quickstart

<!-- TODO: from here. -->

We use `torch.hub` to make loading the model easy -- no cloning of the repo needed. The steps to perform inference are simple:

1. **Install dependancies**: we have 3 inference dependencies only `torch`, `torchaudio`, and `numpy`. Python must be at version 3.10 or greater, and torch must be v2.0 or greater.

2. **Load models**: load the WavLM encoder and HiFiGAN vocoder:

```python
import torch, torchaudio

knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True)
# Or, if you would like the vocoder trained not using prematched data, set prematched=False.
```
3. **Compute features** for input and reference audio:

```python
src_wav_path = '<path to arbitrary 16kHz waveform>.wav'
ref_wav_paths = ['<path to arbitrary 16kHz waveform from target speaker>.wav', '<path to 2nd utterance from target speaker>.wav', ...]

query_seq = knn_vc.get_features(src_wav_path)
matching_set = knn_vc.get_matching_set(ref_wav_paths)
```

4. **Perform the kNN matching and vocoding**:

```python
out_wav = knn_vc.match(query_seq, matching_set, topk=4)
# out_wav is (T,) tensor converted 16kHz output wav using k=4 for kNN.
```

That's it! These default settings provide pretty good results, but feel free to modify the kNN `topk` or use the non-prematched vocoder.
Note: the target speaker from `ref_wav_paths` _can be anything_, but should be clean speech from the desired speaker. The longer the cumulative duration of all reference waveforms, the better the quality will be (but the slower it will take to run). The improvement in quality diminishes beyond 5 minutes of reference speech.

## Checkpoints

<!-- TODO: after this is already done. -->

## Acknowledgements

Parts of code for this project are adapted from the following repositories -- please make sure to check them out! Thank you to the authors of:

- TransFusion: [https://github.com/RF5/transfusion-asr](https://github.com/RF5/transfusion-asr)
- Multinormial diffusion: [https://github.com/ehoogeboom/multinomial_diffusion](https://github.com/ehoogeboom/multinomial_diffusion)
- Mistral-src: [https://github.com/mistralai/mistral-src](https://github.com/mistralai/mistral-src)
- minbpe: [https://github.com/karpathy/minbpe](https://github.com/karpathy/minbpe)
- gemalo-ai's encodec Vocos: [https://github.com/gemelo-ai/vocos](https://github.com/gemelo-ai/vocos)
- librosa for their `.trim()` code: [https://librosa.org/doc/main/generated/librosa.effects.trim.html](https://librosa.org/doc/main/generated/librosa.effects.trim.html)