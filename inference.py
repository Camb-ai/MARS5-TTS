import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Dict, Type, Union
from pathlib import Path
from dataclasses import dataclass
import io

from mars5.model import CodecLM, ResidualTransformer
from vocos import Vocos
from encodec import EncodecModel
from mars5.diffuser import MultinomialDiffusion, DSH, perform_simple_inference
from mars5.minbpe.regex import RegexTokenizer, GPT4_SPLIT_PATTERN
from mars5.minbpe.codebook import CodebookTokenizer
from mars5.ar_generate import ar_generate
from mars5.utils import nuke_weight_norm, construct_padding_mask
from mars5.trim import trim
from safetensors import safe_open
from huggingface_hub import ModelHubMixin, hf_hub_download
import logging

@dataclass
class InferenceConfig():
    """ The defaults configuration variables for TTS inference. """

    ## >>>> AR CONFIG
    # temperature influences probability distribution of logits
    # How to set this variable: high temperatures (T>1) favour less probable outputs while low temperatures reduce randomness
    temperature: float = 0.7

    # Used for sampling - Keeps tokens with the highest probabilities until a certain number (top_k) is reached
    top_k: int = 200 # 0 disables it
    # Used for sampling - keep the top tokens with cumulative probability >= top_p
    top_p: float = 0.2 # 1.0 disables it

    typical_p: float = 1.0
    freq_penalty: float = 3 # increasing it would penalize the model more for repetitions
    presence_penalty: float = 0.4 # increasing it would increase token diversity
    rep_penalty_window: int = 80 # how far in the past to consider when penalizing repetitions. Equates to 5s

    eos_penalty_decay: float = 0.5 # how much to penalize <eos>
    eos_penalty_factor: float = 1 # overal penalty weighting
    eos_estimated_gen_length_factor: float = 1.0 # multiple of len(text_phones) to assume an approximate output length is

    ## >>>> NAR CONFIG
    # defaults, that can be overridden with user specified inputs
    timesteps: int = 200
    x_0_temp: float = 0.7
    q0_override_steps: int = 20 # number of diffusion steps where NAR L0 predictions overrides AR L0 predictions.
    nar_guidance_w: float = 3
    
    max_prompt_dur: float = 12 # maximum length prompt is allowed, in seconds.

    # Maximum AR codes to generate in 1 inference. 
    # Default of -1 leaves it same as training time max AR tokens.
    # Typical values up to ~2x training time can be tolerated, 
    # with ~1.5x trianing time tokens having still mostly ok performance.
    generate_max_len_override: int = -1

    # Whether to deep clone from the reference.
    # Pros: improves intelligibility and speaker cloning performance.
    # Cons: requires reference transcript, and inference takes a bit longer.
    deep_clone: bool = True

    # kv caching helps with optimizing inference speed.
    # disabling/enabling kv caching won't affect output quality
    use_kv_cache: bool = True


    # Leading and trailing silences will be trimmed from final output
    # Trim_db is the threshold (in decibels) below reference to consider as silence
    trim_db: float = 27
    beam_width: int = 1 # only beam width of 1 is currently supported

    ref_audio_pad: float = 0

class Mars5TTS(nn.Module, ModelHubMixin):
    def __init__(self, ar_ckpt, nar_ckpt, device: str = None) -> None:
        super().__init__()

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        self.codec = EncodecModel.encodec_model_24khz().to(device).eval()
        self.codec.set_target_bandwidth(6.0)

        # save and load text tokenize
        self.texttok = RegexTokenizer(GPT4_SPLIT_PATTERN)
        texttok_data = io.BytesIO(ar_ckpt['vocab']['texttok.model'].encode('utf-8'))
        self.texttok.load(texttok_data)

        # save and load speech tokenizer
        self.speechtok = CodebookTokenizer(GPT4_SPLIT_PATTERN)
        speechtok_data = io.BytesIO(ar_ckpt['vocab']['speechtok.model'].encode('utf-8'))
        self.speechtok.load(speechtok_data)
        
        # keep track of tokenization things. 
        self.n_vocab = len(self.texttok.vocab) + len(self.speechtok.vocab)
        self.n_text_vocab = len(self.texttok.vocab) + 1 
        self.diffusion_n_classes: int = 1025 # 1 for padding idx
        # load AR model
        self.codeclm = CodecLM(n_vocab=self.n_vocab, dim=1536, dim_ff_scale=7/3)
        self.codeclm.load_state_dict(ar_ckpt['model'])
        self.codeclm = self.codeclm.to(self.device).eval()
        # load NAR model
        self.codecnar = ResidualTransformer(n_text_vocab=self.n_text_vocab, n_quant=self.diffusion_n_classes, 
                                            p_cond_drop=0, dropout=0)
        self.codecnar.load_state_dict(nar_ckpt['model'])
        self.codecnar = self.codecnar.to(self.device).eval()
        self.default_T = 200

        self.sr = 24000
        self.latent_sr = 75

        # load vocoder
        self.vocos = Vocos.from_pretrained("charactr/vocos-encodec-24khz").to(self.device).eval()
        nuke_weight_norm(self.codec)
        nuke_weight_norm(self.vocos)

    @classmethod
    def _from_pretrained(
        cls: Type["Mars5TTS"],
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        local_files_only: bool,
        token: Optional[Union[str, bool]],
        device: str = None,
        **model_kwargs,
    ) -> "Mars5TTS":
        # Download files from Hub
        logging.info(f">>>>> Downloading AR model")
        ar_ckpt_path = hf_hub_download(repo_id=model_id, filename="mars5_ar.safetensors", revision=revision, cache_dir=cache_dir, force_download=force_download, proxies=proxies, local_files_only=local_files_only, token=token)
        logging.info(f">>>>> Downloading NAR model")
        nar_ckpt_path = hf_hub_download(repo_id=model_id, filename="mars5_nar.safetensors", revision=revision, cache_dir=cache_dir, force_download=force_download, proxies=proxies, local_files_only=local_files_only, token=token)

        ar_ckpt = {}
        with safe_open(ar_ckpt_path, framework='pt', device='cpu') as f:
            metadata = f.metadata()
            ar_ckpt['vocab'] = {'texttok.model': metadata['texttok.model'], 'speechtok.model': metadata['speechtok.model']}
            ar_ckpt['model'] = {}
            for k in f.keys(): ar_ckpt['model'][k] = f.get_tensor(k)
        nar_ckpt = {}
        with safe_open(nar_ckpt_path, framework='pt', device='cpu') as f:
            metadata = f.metadata()
            nar_ckpt['vocab'] = {'texttok.model': metadata['texttok.model'], 'speechtok.model': metadata['speechtok.model']}
            nar_ckpt['model'] = {}
            for k in f.keys(): nar_ckpt['model'][k] = f.get_tensor(k)


        # Init
        return cls(ar_ckpt=ar_ckpt, nar_ckpt=nar_ckpt, device=device)

    @torch.inference_mode
    def vocode(self, tokens: Tensor) -> Tensor:
        """ Vocodes tokens of shape (seq_len, n_q) """
        tokens = tokens.T.to(self.device)
        features = self.vocos.codes_to_features(tokens)
        # A cool hidden feature of vocos vocoding: 
        # setting the bandwidth below to 1 (corresponding to 3 kbps)
        # actually still works on 6kbps input tokens, but *smooths* the output
        # audio a bit, which can help improve quality if its a bit noisy.
        # Hence we use [1] and not [2] below. 
        bandwidth_id = torch.tensor([1], device=self.device)  # 6 kbps
        wav_diffusion = self.vocos.decode(features, bandwidth_id=bandwidth_id)
        return wav_diffusion.cpu().squeeze()[None]

    @torch.inference_mode
    def get_speaker_embedding(self, ref_audio: Tensor) -> Tensor:
        """ Given `ref_audio` (bs, T) audio tensor, compute the implicit speakre embedding of shape (bs, dim). """
        if ref_audio.dim() == 1: ref_audio = ref_audio[None]
        spk_reference = self.codec.encode(ref_audio[None].to(self.device))[0][0]
        spk_reference = spk_reference.permute(0, 2, 1)
        bs = spk_reference.shape[0]
        if bs != 1:
            raise AssertionError(f"Speaker embedding extraction only implemented using for bs=1 currently.")
        spk_seq = self.codeclm.ref_chunked_emb(spk_reference) # (bs, sl, dim)
        spk_ref_emb = self.codeclm.spk_identity_emb.weight[None].expand(bs, -1, -1) # (bs, 1, dim)

        spk_seq = torch.cat([spk_ref_emb, spk_seq], dim=1) # (bs, 1+sl, dim)
        # add pos encoding
        spk_seq = self.codeclm.pos_embedding(spk_seq)
        # codebook goes from indices 0->1023, padding is idx 1024 (the 1025th entry)
        src_key_padding_mask = construct_padding_mask(spk_reference[:, :, 0], 1024) 
        src_key_padding_mask = torch.cat((
                                            # append a zero here since we DO want to attend to initial position.
                                            torch.zeros(src_key_padding_mask.shape[0], 1, dtype=bool, device=src_key_padding_mask.device), 
                                            src_key_padding_mask
                                            ), 
                                            dim=1)
        # pass through transformer
        res = self.codeclm.spk_encoder(spk_seq, is_causal=False, src_key_padding_mask=src_key_padding_mask)[:, :1] # select first element -> now (bs, 1, dim).
        return res.squeeze(1)

    @torch.inference_mode
    def tts(self, text: str, ref_audio: Tensor, ref_transcript: Optional[str] = None, 
            cfg: Optional[InferenceConfig] = InferenceConfig()) -> Tensor:
        """ Perform TTS for `text`, given a reference audio `ref_audio` (of shape [sequence_length,], sampled at 24kHz) 
        which has an associated `ref_transcript`. Perform inference using the inference 
        config given by `cfg`, which controls the temperature, top_p, etc...
        Returns:
        - `ar_codes`: (seq_len,) long tensor of discrete coarse code outputs from the AR model.
        - `out_wav`: (T,) float output audio tensor sampled at 24kHz.
        """

        if cfg.deep_clone and ref_transcript is None:
            raise AssertionError(
                ("Inference config deep clone is set to true, but reference transcript not specified! "
                "Please specify the transcript of the prompt, or set deep_clone=False in the inference `cfg` argument."
            ))
        ref_dur = ref_audio.shape[-1]/self.sr
        if ref_dur > cfg.max_prompt_dur:
            logging.warning((f"Reference audio duration is {ref_dur:.2f} > max suggested ref audio. "
                            f"Expect quality degradations. We recommend you trim prompt to be shorter than max prompt length."))

        # get text codes. 
        text_tokens =  self.texttok.encode("<|startoftext|>"+text.strip()+"<|endoftext|>", 
                                           allowed_special='all')

        text_tokens_full = self.texttok.encode("<|startoftext|>"+ ref_transcript + ' ' + str(text).strip()+"<|endoftext|>", 
                                            allowed_special='all')
        
        if ref_audio.dim() == 1: ref_audio = ref_audio[None]
        if ref_audio.shape[0] != 1: ref_audio = ref_audio.mean(dim=0, keepdim=True)
        ref_audio = F.pad(ref_audio, (int(self.sr*cfg.ref_audio_pad), 0))
        # get reference audio codec tokens
        prompt_codec = self.codec.encode(ref_audio[None].to(self.device))[0][0] # (bs, n_q, seq_len)

        n_speech_inp = 0
        n_start_skip = 0
        q0_str = ' '.join([str(t) for t in prompt_codec[0, 0].tolist()])
        # Note, in the below, we do NOT want to encode the <eos> token as a part of it, since we will be continuing it!!!
        speech_tokens = self.speechtok.encode(q0_str.strip()) # + "<|endofspeech|>", allowed_special='all')
        spk_ref_codec = prompt_codec[0, :, :].T # (seq_len, n_q)

        raw_prompt_acoustic_len = len(prompt_codec[0,0].squeeze())
        offset_speech_codes = [p+len(self.texttok.vocab) for p in speech_tokens]
        if not cfg.deep_clone: 
            # shallow clone, so 
            # 1. clip existing speech codes to be empty (n_speech_inp = 0)
            offset_speech_codes = offset_speech_codes[:n_speech_inp]
        else: 
            # Deep clone, so
            # 1. set text to be text of prompt + target text
            text_tokens = text_tokens_full
            # 2. update n_speech_inp to be length of prompt, so we only display from ths `n_speech_inp` onwards in the final output.
            n_speech_inp = len(offset_speech_codes)
        prompt = torch.tensor(text_tokens + offset_speech_codes, dtype=torch.long, device=self.device)
        first_codec_idx = prompt.shape[-1] - n_speech_inp + 1

        # ---> perform AR code generation
        logging.debug(f"Raw acoustic prompt length: {raw_prompt_acoustic_len}")

        ar_codes = ar_generate(self.texttok, self.speechtok, self.codeclm, 
                               prompt, spk_ref_codec, first_codec_idx, 
                               max_len=cfg.generate_max_len_override if cfg.generate_max_len_override > 1 else 2000,
                               fp16=True if torch.cuda.is_available() else False,
                               temperature=cfg.temperature, topk=cfg.top_k, top_p=cfg.top_p, typical_p=cfg.typical_p,
                               alpha_frequency=cfg.freq_penalty, alpha_presence=cfg.presence_penalty, penalty_window=cfg.rep_penalty_window,
                               eos_penalty_decay=cfg.eos_penalty_decay, eos_penalty_factor=cfg.eos_penalty_factor,
                               beam_width=cfg.beam_width, beam_length_penalty=1,
                               n_phones_gen=round(cfg.eos_estimated_gen_length_factor*len(text)), 
                               vocode=False, use_kv_cache=cfg.use_kv_cache)

        # Parse AR output
        output_tokens = ar_codes - len(self.texttok.vocab)
        output_tokens = output_tokens.clamp(min=0).squeeze()[first_codec_idx:].cpu().tolist()
        gen_codes_decoded = self.speechtok.decode_int(output_tokens)
        gen_codes_decoded = torch.tensor([s for s in gen_codes_decoded if type(s) == int], dtype=torch.long, device=self.device)

        c_text = torch.tensor(text_tokens, dtype=torch.long, device=self.device)[None]
        c_codes = prompt_codec.permute(0, 2, 1)
        c_texts_lengths = torch.tensor([len(text_tokens)], dtype=torch.long, device=self.device)
        c_codes_lengths = torch.tensor([c_codes.shape[1],], dtype=torch.long, device=self.device)

        _x = gen_codes_decoded[None, n_start_skip:, None].repeat(1, 1, 8) # (seq_len) -> (1, seq_len, 8)
        x_padding_mask = torch.zeros((1, _x.shape[1]), dtype=torch.bool, device=_x.device)

        # ---> perform DDPM NAR inference
        T = self.default_T
        diff = MultinomialDiffusion(self.diffusion_n_classes, timesteps=T, device=self.device)

        dsh_cfg = DSH(last_greedy=True, x_0_temp=cfg.x_0_temp, 
                        guidance_w=cfg.nar_guidance_w, 
                        deep_clone=cfg.deep_clone, jump_len=1, jump_n_sample=1, 
                        q0_override_steps=cfg.q0_override_steps,
                        enable_kevin_scaled_inference=True, # see TransFusion ASR for explanation of this
                        progress=False)
        
        final_output = perform_simple_inference(self.codecnar,(
            c_text, c_codes, c_texts_lengths, c_codes_lengths, _x, x_padding_mask
        ), diff, diff.num_timesteps, torch.float16, dsh=dsh_cfg, retain_quant0=True) # (bs, seq_len, n_quant)

        skip_front = raw_prompt_acoustic_len if cfg.deep_clone else 0 
        final_output = final_output[0, skip_front:].to(self.device)  # (seq_len, n_quant)

        # vocode final output and trim silences
        final_audio = self.vocode(final_output).squeeze()
        final_audio, _ = trim(final_audio.cpu(), top_db=cfg.trim_db)

        return gen_codes_decoded, final_audio
