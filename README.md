# MARS5: A novel speech model for insane prosody
<div id="top" align="center">

   ![MARS5 Banner](assets/github-banner.png)

   <h3>
   <a href="https://www.loom.com/share/a6e7c6658f9f4b09a696926a98dd6fcc"> Why MARS5? </a> |
   <a href="https://github.com/Camb-ai/MARS5-TTS/blob/master/docs/architecture.md"> Model Architecture </a> |
   <a href="https://179c54d254f7.ngrok.app/"> Samples </a> |
   <a href="https://camb.ai/"> Camb AI Website </a></h3>

   [![GitHub Repo stars](https://img.shields.io/github/stars/Camb-ai/MARS5-TTS?style=social)](https://github.com/Camb-ai/MARS5-TTS/stargazers)
   [![Join our Discord](https://discordapp.com/api/guilds/1107565548864290840/widget.png)](https://discord.gg/FFQNCSKSXX)
   [![HuggingFace badge](https://img.shields.io/badge/%F0%9F%A4%97HuggingFace-Join-yellow)](https://huggingface.co/CAMB-AI/MARS5-TTS)
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Camb-ai/mars5-tts/blob/master/mars5_demo.ipynb)


</div>





# Approach

This is the repo for the MARS5 English speech model (TTS) from CAMB.AI.

The model follows a two-stage AR-NAR pipeline with a distinctively novel NAR component (see more info in the [Architecture](docs/architecture.md)).

With just 5 seconds of audio and a snippet of text, MARS5 can generate speech even for prosodically hard and diverse scenarios like sports commentary, anime and more. Check out our demo:




https://github.com/Camb-ai/MARS5-TTS/assets/23717819/3e191508-e03c-4ff9-9b02-d73ae0ebefdd


Watch full video here: [![Youtube](https://img.shields.io/badge/YouTube-red)](https://www.youtube.com/watch?v=bmJSLPYrKtE)

![Mars 5 simplified diagram](docs/assets/MARS5_Architecture.png)

**Figure**: The high-level architecture flow of MARS5. Given text and a reference audio, coarse (L0) encodec speech features are obtained through an autoregressive transformer model. Then, the text, reference, and coarse features are refined in a multinomial DDPM model to produce the remaining encodec codebook values. The output of the DDPM is then vocoded to produce the final audio.

Because the model is trained on raw audio together with byte-pair-encoded text, it can be steered with things like punctuation and capitalization.
E.g. To add a pause, add a comma to that part in the transcript. Or, to emphasize a word, put it in capital letters in the transcript.
This enables a fairly natural way for guiding the prosody of the generated output.

Speaker identity is specified using an audio reference file between 2-12 seconds, with lengths around 6s giving optimal results.
Further, by providing the transcript of the reference, MARS5 enables one to do a '_deep clone_' which improves the quality of the cloning and output, at the cost of taking a bit longer to produce the audio.
For more details on this and other performance and model details, please see the [docs folder](docs/architecture.md).

## Comparison with ElevenLabs and Metavoice

| Example | Text | Reference Voice | Elevenlabs Output | Metavoice Output | Mars 5 Output |
|---------|------|-----------------|-------------------|------------------|---------------|
| 1       | I just can't anymore. | [Reference](demo_egs/comparisons/cant_anymore/reference.wav) | [Elevenlabs](demo_egs/comparisons/cant_anymore/el.wav) | [Metavoice](demo_egs/comparisons/cant_anymore/metavoice.wav) | [Mars 5](demo_egs/comparisons/cant_anymore/mars5.flac) |
| 2       | Have you heard the legend of Mars? | [Reference](demo_egs/comparisons/seen_mars/reference.wav) | [Elevenlabs](demo_egs/comparisons/seen_mars/el.wav) | [Metavoice](demo_egs/comparisons/seen_mars/metavoice.wav) | [Mars 5](demo_egs/comparisons/seen_mars/mars.flac) |

<details>
<summary>Additional Samples</summary>

| Example | Text | Reference Voice | Output |
|---------|------|-----------------|--------|
| 1       | Ladies and gentlemen, now is the time for the next step in speech synthesis | [Reference](demo_egs/eg1_ref.flac) | [Output](demo_egs/eg1_next_step_in_speech.flac) |
| 2       | Let us introduce to you our new text to speech model, Mars five. | [Reference](demo_egs/eg2_ref.flac) | [Output](demo_egs/eg2_introduce.flac) |
| 3       | Mars is quite... insane. | [Reference](demo_egs/eg3_ref.wav) | [Output](demo_egs/eg3_mars_is_amazing.flac) |
| 4       | Mars can be a bit quirky at times. | [Reference](demo_egs/eg4_ref.wav) | [Output](demo_egs/eg4_quirky2.flac) |
| 5       | And talk back with a little bit of spirit in it. | [Reference](demo_egs/eg5_ref.wav) | [Output](demo_egs/eg5_talk_back_annoyance.flac) |
| 6       | You should speak! | [Reference](demo_egs/eg6_ref.wav) | [Output](demo_egs/eg6_you_should_speak_theseus.flac) |
| 7       | On his rump, he hoists the game winning shot! | [Reference](demo_egs/eg7_ref.wav) | [Output](demo_egs/eg7_sports.flac) |
| 8       | Can he make it in this shot? GO!! | [Reference](demo_egs/eg8_ref.wav) | [Output](demo_egs/eg8_shout_ff.wav) |
| 9       | Then the wizard said softly, "don't make a sound". | [Reference](demo_egs/eg9_ref.wav) | [Output](demo_egs/eg9_softly.flac) |
| 10      | I'm gonna make him an offer he can't refuse | [Reference](demo_egs/eg11_ref.wav) | [Output](demo_egs/eg11_bob_ross.flac) |
| 11      | Houston, we have a problem. | [Reference](demo_egs/eg12_ref.wav) | [Output](demo_egs/eg12_bob_ross_output_2.flac) |
| 12      | There appears to be a snake in my boot. | [Reference](demo_egs/eg13_ref.flac) | [Output](demo_egs/eg13_snake_in_my_boot1.flac) |
| 13      | I'm gonna make him an offer he can't refuse. | [Reference](demo_egs/eg14_ref.wav) | [Output](demo_egs/eg14_micki_mouse_output_1.flac) |
| 14      | We hope you enjoyed the demo, feel free to try generate some samples yourself below. | [Reference](demo_egs/eg18_ref.wav) | [Output](demo_egs/eg18_finale.flac) |

</details>


## Quick links

- [CAMB.AI website](https://camb.ai/) (access MARS5 in 140+ languages for TTS and dubbing)
- Technical details and architecture: [in the docs folder](docs/architecture.md)
- Colab quickstart: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Camb-ai/mars5-tts/blob/master/mars5_demo.ipynb)
- Sample page with a few hard prosodic samples: [https://camb-ai.github.io/MARS5-TTS/](https://camb-ai.github.io/MARS5-TTS/)
- Online demo: [here](https://179c54d254f7.ngrok.app/)


## Quickstart


We use `torch.hub` to make loading the model easy -- no cloning of the repo needed. The steps to perform inference are simple:

1. **Installation using pip**:

    Requirements:
    - Python >= 3.10
    - Torch >= 2.0
    - Torchaudio
    - Librosa
    - Vocos
    - Encodec

```bash
pip install --upgrade torch torchaudio librosa vocos encodec
```

2. **Load models**: load the MARS5 AR and NAR model from torch hub:

```python
import torch, librosa

mars5, config_class = torch.hub.load('Camb-ai/mars5-tts', 'mars5_english', trust_repo=True)
# The `mars5` contains the AR and NAR model, as well as inference code.
# The `config_class` contains tunable inference config settings like temperature.
```
3. **Pick a reference** and optionally its transcript:

```python
# Load reference audio between 1-12 seconds.
wav, sr = librosa.load('<path to arbitrary 24kHz waveform>.wav',
                       sr=mars5.sr, mono=True)
wav = torch.from_numpy(wav)
ref_transcript = "<transcript of the reference audio>"
```

*Note: The reference transcript is optional. Pass it if you wish to do a deep clone.*

MARS5 supports 2 kinds of inference: a shallow, fast inference whereby you do not need the transcript of the reference (we call this a _shallow clone_), and a second slower, but typically higher quality way, which we call a _deep clone_.
To use the deep clone, you need the prompt transcript. See the [model architecture](docs/architecture.md) for more info on this.

4. **Perform the synthesis**:

```python
# Pick whether you want a deep or shallow clone. Set to False if you don't know prompt transcript or want fast inference. Set to True if you know transcript and want highest quality.
deep_clone = True
# Below you can tune other inference settings, like top_k, temperature, top_p, etc...
cfg = config_class(deep_clone=deep_clone, rep_penalty_window=100,
                      top_k=100, temperature=0.7, freq_penalty=3)

ar_codes, output_audio = mars5.tts("The quick brown rat.", wav,
          ref_transcript,
          cfg=cfg)
# output_audio is (T,) shape float tensor corresponding to the 24kHz output audio.
```

**That's it!** These default settings provide pretty good results, but feel free to tune the inference settings to optimize the output for your particular usecase. See the [`InferenceConfig`](inference.py) code or the demo notebook for info and docs on all the different inference settings.

_Some tips for best quality:_
- Make sure reference audio is clean and between 1 second and 12 seconds.
- Use deep clone and provide an accurate transcript for the reference.
- Use proper punctuation -- the model can be guided and made better or worse with proper use of punctuation and capitalization.


## Model Details

**Checkpoints**

The checkpoints for MARS5 are provided under the releases tab of this github repo. We provide two checkpoints:

- AR fp16 checkpoint [~750M parameters], along with config embedded in the checkpoint.
- NAR fp16 checkpoint [~450M parameters], along with config embedded in the checkpoint.
- The byte-pair encoding tokenizer used for the L0 encodec codes and the English text is embedded in each checkpoint under the `'vocab'` key, and follows roughly the same format of a saved minbpe tokenizer.

**Hardware Requirements**:

You must be able to store at least 750M+450M params on GPU, and do inference with 750M of active parameters. In general, at least **20GB of GPU VRAM** is needed to run the model on GPU (we plan to further optimize this in the future).

If you do not have the necessary hardware requirements and just want to use MARS5 in your applications, you can use it via our [API](https://docs.camb.ai/). If you need some extra credits to test it for your use case, feel free to reach out to `help@camb.ai`.

## Roadmap

MARS5 is not perfect at the moment, and we are working on improving its quality, stability, and performance.
Rough areas we are looking to improve, and welcome any contributions in:

☐ Improving inference stability and consistency <br />
☐ Speed/performance optimizations <br />
☐ Improving reference audio selection when given long references. <br />
☐ Benchmark performance numbers for MARS5 on standard speech datasets.

If you would like to contribute any improvement to MARS5, please feel free to contribute (guidelines below).


## Contributions

We welcome any contributions to improving the model. As you may find when experimenting, it can produce really great results, it can still be further improved to create excellent outputs _consistently_.
We'd also love to see how you used MARS5 in different scenarios, please use the [🙌 Show and tell](https://github.com/Camb-ai/MARS5-TTS/discussions/categories/show-and-tell) category in Discussions to share your examples.

**Contribution format**:

The preferred way to contribute to our repo is to fork the [master repository](https://github.com/Camb-ai/mars5-tts) on GitHub:

1. Fork the repo on github
2. Clone the repo, set upstream as this repo: `git remote add upstream git@github.com:Camb-ai/mars5-tts.git`
3. Make a new local branch and make your changes, commit changes.
4. Push changes to new upstream branch: `git push --set-upstream origin <NAME-NEW-BRANCH>`
5. On github, go to your fork and click 'Pull Request' to begin the PR process. Please make sure to include a description of what you did/fixed.

## License

We are open-sourcing MARS5 in English under GNU AGPL 3.0, but you can request to use it under a different license by emailing help@camb.ai.

## Join Our Team

We're an ambitious team, globally distributed, with a singular aim of making everyone's voice count. At CAMB.AI, we're a research team of Interspeech-published, Carnegie Mellon, ex-Siri engineers and we're looking for you to join our team.

We're actively hiring; please drop us an email at ack@camb.ai if you're interested. Visit our [careers page](https://www.camb.ai/careers) for more info.


## Community

Join CAMB.AI community on [Forum](https://github.com/Camb-ai/MARS5-TTS/discussions) and
[Discord](https://discord.gg/FFQNCSKSXX) to share any suggestions, feedback, or questions with our team.


## Acknowledgements

Parts of code for this project are adapted from the following repositories -- please make sure to check them out! Thank you to the authors of:

- AWS: For providing much needed compute resources (NVIDIA H100s) to enable training of the model.
- TransFusion: [https://github.com/RF5/transfusion-asr](https://github.com/RF5/transfusion-asr)
- Multinomial diffusion: [https://github.com/ehoogeboom/multinomial_diffusion](https://github.com/ehoogeboom/multinomial_diffusion)
- Mistral-src: [https://github.com/mistralai/mistral-src](https://github.com/mistralai/mistral-src)
- minbpe: [https://github.com/karpathy/minbpe](https://github.com/karpathy/minbpe)
- gemelo-ai's encodec Vocos: [https://github.com/gemelo-ai/vocos](https://github.com/gemelo-ai/vocos)
- librosa for their `.trim()` code: [https://librosa.org/doc/main/generated/librosa.effects.trim.html](https://librosa.org/doc/main/generated/librosa.effects.trim.html)
