import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .nn_future import (FNNSwiGLU, MistralTransformer, ModelArgs,
                        RotatingBufferCache, SinePositionalEmbedding)
from .utils import construct_padding_mask, length_to_mask

LAYERNORM_EPS = 4e-5

# ------------------------
# Code adapted from OpenAI guided diffusion repo

def timestep_embedding(timesteps, dim, max_period=10000, dtype=torch.float32):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(dtype)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# --------------------------------
# autoregressive codec language model


class CodecLM(nn.Module):

    def __init__(self, n_vocab, dim=1536, nhead=24, n_layers=26, n_spk_layers=2, dim_ff_scale=None, sliding_window=3000) -> None:
        super().__init__()

        if dim_ff_scale is None: hidden_dim = int(dim*4*(3/4))
        else: hidden_dim = int(dim*dim_ff_scale)

        self.cfg = ModelArgs(n_vocab, dim=dim, n_layers=n_layers, n_heads=nhead, n_kv_heads=nhead, hidden_dim=hidden_dim, sliding_window=sliding_window)
        self.ar = MistralTransformer(self.cfg)

        self.embed = nn.Embedding(n_vocab, dim)

        # --- spk embedding network
        dim_ff = int(dim*4*(3/4))
        self.pos_embedding = SinePositionalEmbedding(dim, scale=False, alpha=True)
        self.ref_chunked_emb = ChunkedEmbedding(1024 + 1, 8, dim) # add 1 for pad idx
        self.spk_identity_emb = nn.Embedding(1, dim)
        # define custom decoder
        encoder_layer = nn.TransformerEncoderLayer(dim, nhead, dim_ff,
                                                activation=FNNSwiGLU(dim, dim_ff), dropout=0,
                                                batch_first=True, norm_first=True, layer_norm_eps=LAYERNORM_EPS)
        encoder_layer.linear1 = nn.Identity()
        self.spk_encoder = nn.TransformerEncoder(encoder_layer, n_spk_layers, norm=nn.LayerNorm(dim, eps=LAYERNORM_EPS))
        # monkeypatch for broken copy.deepcopy of nn.Modules in nn.TransformerDecoder
        for l in self.spk_encoder.layers: l.activation = FNNSwiGLU(dim, dim_ff)


    @torch.inference_mode
    def get_spk_embedding(self, spk_reference, c_codes_lengths=None) -> Tensor:
        """ Gets speaker reference embeddings using `spk_reference` codes of shape (bs, seq_len, n_codebooks). """
        bs = spk_reference.shape[0]
        if bs != 1:
            raise AssertionError(f"Speaker embedding extraction only implemented using for bs=1 currently.")
        spk_seq = self.ref_chunked_emb(spk_reference) # (bs, sl, dim)
        spk_ref_emb = self.spk_identity_emb.weight[None].expand(bs, -1, -1) # (bs, 1, dim)

        spk_seq = torch.cat([spk_ref_emb, spk_seq], dim=1) # (bs, 1+sl, dim)
        # add pos encoding
        spk_seq = self.pos_embedding(spk_seq)
        # codebook goes from indices 0->1023, padding is idx 1024 (the 1025th entry)
        src_key_padding_mask = construct_padding_mask(spk_reference[:, :, 0], 1024) 
        src_key_padding_mask = torch.cat((
                                            # append a zero here since we DO want to attend to initial position.
                                            torch.zeros(src_key_padding_mask.shape[0], 1, dtype=bool, device=src_key_padding_mask.device), 
                                            src_key_padding_mask
                                            ), 
                                            dim=1)
        # pass through transformer
        res = self.spk_encoder(spk_seq, is_causal=False, src_key_padding_mask=src_key_padding_mask)[:, :1] # select first element -> now (bs, 1, dim).
        return res.squeeze(1)


    def forward(self, x: Tensor, x_padding_mask: Optional[Tensor] = None, spk_reference: Optional[Tensor] = None,
                cache: Optional[RotatingBufferCache] = None, counter: int = 0) -> Tensor:
        """ Inputs:
            - `x`: (bs, seq_len, vocab_size) 
            - `x_padding_mask`: (bs, seq_len) mask for each input, True for positions to *ignore*, False otherwise.
                Note that since this is an autoregressive model, this doesn't actually matter for infernece, so it is ignored at inference. 
            - `spk_reference`: (bs, seq_len, n_codebooks) corresponding to the speaker reference to clone from.
            - `cache` and `counter`: used for kv caching, optional.

            Returns `x` of same shape (bs, seq_len, dim)
        """
        x = self.embed(x)

        # --- speaker reference/embedding
        if spk_reference is not None:
            # compute ref
            bs = spk_reference.shape[0]
            spk_seq = self.ref_chunked_emb(spk_reference) # (bs, sl, dim)
            spk_ref_emb = self.spk_identity_emb.weight[None].expand(bs, -1, -1) # (bs, 1, dim)

            spk_seq = torch.cat([spk_ref_emb, spk_seq], dim=1) # (bs, 1+sl, dim)
            # add pos encoding
            spk_seq = self.pos_embedding(spk_seq)
            # codebook goes from indices 0->1023, padding is idx 1024 (the 1025th entry)
            src_key_padding_mask = construct_padding_mask(spk_reference[:, :, 0], 1024) 
            src_key_padding_mask = torch.cat((
                                                # append a zero here since we DO want to attend to initial position.
                                                torch.zeros(src_key_padding_mask.shape[0], 1, dtype=bool, device=src_key_padding_mask.device), 
                                                src_key_padding_mask
                                             ), 
                                             dim=1)
            # pass through transformer
            res = self.spk_encoder(spk_seq, is_causal=False, src_key_padding_mask=src_key_padding_mask)[:, :1] # select first element -> now (bs, 1, dim).
            
            x = torch.cat([res, x], dim=1)

        positions = torch.arange(0, x.shape[1], device=x.device, dtype=torch.long)
        if cache is not None and counter != 1:
            # using only the last token to predict the next one
            x = x[:,-1,:].unsqueeze(1)
            positions = positions[-1:]

        x = self.ar(x, positions, cache) # (bs, seq_len, vocab)
        if spk_reference is not None and (cache is None or counter == 1):
            x = x[:, 1:] # strip out the first output token corresponding to the speaker embedding token.

        return x


# -------------------------
# residual discrete diffusion model

class ChunkedEmbedding(nn.Module):

    def __init__(self, codebook_size: int, n_quantizer: int, dim: int) -> None:
        super().__init__()
        assert dim % n_quantizer == 0, f"ChunkedEmbedding output dim ({dim}) must be divisible by n_quant {n_quantizer}"
        self.embs = nn.ModuleList([nn.Embedding(codebook_size, dim//n_quantizer) for _ in range(n_quantizer)])

    def forward(self, x: Tensor) -> Tensor:
        """ Embeds each codebook index in `x` (bs, seq_len, n_quantizer) to an embedding vector, concatenating results.
        Returns output of shape (bs, seq_len, dim)
        """
        y = torch.cat([self.embs[i](x[..., i]) for i in range(x.shape[-1])], dim=-1)
        return y



class ResidualTransformer(nn.Module):

    def __init__(self, n_text_vocab, n_quant=1024, dim=1024, nhead=16, 
                 enc_layers=8, dec_layers=16, n_spk_layers=3,
                 c_quant_levels=8, pred_quant_levels=8, 
                 t_emb_dim=1024, norm_first=True, p_cond_drop=0.1, dropout=0) -> None:
        super().__init__()

        self.cond_pos_embedding = SinePositionalEmbedding(dim, scale=False, alpha=True)
        self.pos_embedding = SinePositionalEmbedding(dim, scale=False, alpha=True)

        # *4 from heuristic, *2/3 from swiglu, since there are 3 linear matrices not 2.
        # so we must keep # params the same.
        dim_ff = int(dim*4*(3/4))

        # define custom encoder
        encoder_layer = nn.TransformerEncoderLayer(dim, nhead, dim_ff,
                            activation=FNNSwiGLU(dim, dim_ff), dropout=dropout,
                            batch_first=True, norm_first=norm_first, layer_norm_eps=LAYERNORM_EPS)
        encoder_layer.linear1 = nn.Identity()
        encoder = nn.TransformerEncoder(encoder_layer, enc_layers, norm=nn.LayerNorm(dim, eps=LAYERNORM_EPS) if norm_first else None)

        # define custom decoder
        decoder_layer = nn.TransformerDecoderLayer(dim, nhead, dim_ff,
                                                activation=FNNSwiGLU(dim, dim_ff), dropout=dropout,
                                                batch_first=True, norm_first=norm_first, layer_norm_eps=LAYERNORM_EPS)
        decoder_layer.linear1 = nn.Identity()
        decoder = nn.TransformerDecoder(decoder_layer, dec_layers, norm=nn.LayerNorm(dim, eps=LAYERNORM_EPS) if norm_first else None)

        # monkeypatch for broken copy.deepcopy of nn.Modules in nn.TransformerDecoder
        for l in decoder.layers: l.activation = FNNSwiGLU(dim, dim_ff)

        self.tfm = nn.Transformer(dim, nhead, dim_feedforward=dim_ff, batch_first=True, 
            norm_first=norm_first,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            custom_encoder=encoder,
            custom_decoder=decoder,
            layer_norm_eps=LAYERNORM_EPS,
            dropout=dropout
        )
        # Timestep embedding network
        self.t_emb_dim = t_emb_dim
        self.timestep_encoder_emb = nn.Sequential(
            nn.Linear(t_emb_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.timestep_decoder_emb = nn.Sequential(
            nn.Linear(t_emb_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

        self.text_embed = nn.Embedding(n_text_vocab, dim)

        ## ----> reference / conditioning encoder:
        self.ref_embedder = ChunkedEmbedding(n_quant, c_quant_levels, dim)
        self.ref_pos_embedding = SinePositionalEmbedding(dim, scale=False, alpha=True)
        self.spk_identity_emb = nn.Embedding(1, dim)
        spk_encoder_layer = nn.TransformerEncoderLayer(dim, nhead, dim_ff,
                                                activation=FNNSwiGLU(dim, dim_ff), dropout=dropout,
                                                batch_first=True, norm_first=True, layer_norm_eps=LAYERNORM_EPS)
        spk_encoder_layer.linear1 = nn.Identity()
        self.spk_encoder = nn.TransformerEncoder(spk_encoder_layer, n_spk_layers, norm=nn.LayerNorm(dim, eps=LAYERNORM_EPS))
        # monkeypatch for broken copy.deepcopy of nn.Modules in nn.TransformerDecoder
        for l in self.spk_encoder.layers: l.activation = FNNSwiGLU(dim, dim_ff)
        # ----> end speaker encoder network

        # self.residual_encoder = nn.Embedding(n_quant, dim) # only encode first quantization level of decoder input.
        self.residual_encoder = ChunkedEmbedding(n_quant, c_quant_levels, dim)

        self.residual_decoder = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, n_quant)
            ) for i in range(pred_quant_levels)
        ])
        self.n_quantizer = pred_quant_levels
        self.p_cond_drop = p_cond_drop


    @torch.inference_mode
    def get_spk_embedding(self, c_codes, c_codes_length) -> Tensor:
        """ Obtain speaker embedding vectors using `c_codes` from reference encodec sequences, and `c_codes_length` of lengths for each sequence """
        bs = c_codes.shape[0]
        spk_seq = self.ref_embedder(c_codes) # (bs, sl, dim)
        spk_ref_emb = self.spk_identity_emb.weight[None].expand(bs, -1, -1) # (bs, 1, dim)
        spk_seq = torch.cat([spk_ref_emb, spk_seq], dim=1) # (bs, 1+sl, dim)
        # add pos encoding
        spk_seq = self.ref_pos_embedding(spk_seq)

        # add 1 to c_codes_length to account for the fact that we concatenate the spk_ref_emb to it. 
        src_key_padding_mask = length_to_mask(c_codes_length+1, torch.zeros_like(c_codes_length), max_len=spk_seq.shape[1])
        src_key_padding_mask = src_key_padding_mask.to(dtype=torch.bool, device=spk_seq.device)

        # pass through transformer
        res = self.spk_encoder(spk_seq, is_causal=False, src_key_padding_mask=src_key_padding_mask)[:, :1] # select first element -> now (bs, 1, dim).
        return res.squeeze(1)


    def forward(self, c_text: Tensor, c_codes: Tensor, c_texts_length: Tensor, c_codes_length: Tensor, 
                x: Tensor, x_padding_mask: Tensor, t: Tensor, drop_cond=False):
        """ Input:
            - `c_text`: (bs, seq_len1) the prompt text (BPE encoded)
            - `c_codes`: (bs, seq_len2, n_quant) the full tokenized codes of the reference speech
            - `c_texts_length`: (bs, ) the length of the codes in the text prompt
            - `c_codes_length`: (bs, ) the length of the prompt acoustic token codes in `c_codes`.
            - `x`: (bs, seq_len3) L0 residual codes
            - `x`: (bs, seq_len3, n_quant) L0 residual codes
            - `x_padding_mask`: (bs, seq_len3) masking for residual codes
            - `t`: (bs) timestep
            - `drop_cond`: bool, whether or not to forcibly drop the conditioning information.
        Returns:
            - outs: (bs, seq_len, n_quantizer, codebook_size)
        """
        
        c_text = self.text_embed(c_text) # (bs, seq_len1, dim)

        ## ----> reference / conditioning encoder:
        bs = c_codes.shape[0]

        
        if self.training:
            zero_cond_inds = torch.rand_like(t, dtype=c_text.dtype) < self.p_cond_drop
        else:
            # never randomly zero when in eval mode
            zero_cond_inds = torch.zeros_like(t, dtype=torch.bool)
            if drop_cond:
                # force drop conditioning
                zero_cond_inds = torch.ones_like(t, dtype=torch.bool)
        
        c_codes_length[zero_cond_inds] = 0
        c_codes[zero_cond_inds] = 1024

        spk_seq = self.ref_embedder(c_codes) # (bs, sl, dim)
        spk_ref_emb = self.spk_identity_emb.weight[None].expand(bs, -1, -1) # (bs, 1, dim)
        spk_seq = torch.cat([spk_ref_emb, spk_seq], dim=1) # (bs, 1+sl, dim)
        # add pos encoding
        spk_seq = self.ref_pos_embedding(spk_seq)

        # add 1 to c_codes_length to account for the fact that we concatenate the spk_ref_emb to it. 
        src_key_padding_mask = length_to_mask(c_codes_length+1, torch.zeros_like(c_codes_length), max_len=spk_seq.shape[1])
        src_key_padding_mask = src_key_padding_mask.to(dtype=torch.bool, device=spk_seq.device)

        # pass through transformer
        res = self.spk_encoder(spk_seq, is_causal=False, src_key_padding_mask=src_key_padding_mask)[:, :1] # select first element -> now (bs, 1, dim).
        c_codes = res # (bs, 1, dim)
        c_codes_lengths_extract = torch.ones_like(c_codes_length) # manually override all the code lengths to equal 1, since we only have 1 spk embedding. 
        ## ----> end reference / conditioning encoder:

        ## ----> timestep embeddings and parsing
        t_emb = timestep_embedding(t, self.t_emb_dim, dtype=c_text.dtype)
        t_emb_encoder = self.timestep_encoder_emb(t_emb) # (bs, t_dim)
        t_emb_decoder = self.timestep_decoder_emb(t_emb)
        
        ## ----> concatenating text/phone inputs and implicit speaker embedding. 
        c_phones_unpacked = nn.utils.rnn.unpad_sequence(c_text, c_texts_length.cpu(), batch_first=True)
        c_codes_unpacked = nn.utils.rnn.unpad_sequence(c_codes, c_codes_lengths_extract.cpu(), batch_first=True)
        # >>> Concat [speaker codes, text codes]
        assert all(b.shape[0] == 1 for b in c_codes_unpacked)
        c_joined = [torch.cat((b, a), dim=0) for a, b in zip(c_phones_unpacked, c_codes_unpacked)]

        c = nn.utils.rnn.pad_sequence(c_joined, batch_first=True)
        c_joined_lengths = torch.tensor([p.shape[0] for p in c_joined], device=c.device, dtype=torch.long)
        c_padding_mask = length_to_mask(c_joined_lengths, torch.zeros_like(c_joined_lengths))
        c = self.cond_pos_embedding(c)

        ## Format input:
        x = self.residual_encoder(x) # (bs, seq_len3, dim)

        x = self.pos_embedding(x)

        x = x + t_emb_decoder[:, None]
        c = c + t_emb_encoder[:, None]
        ## Perform prediction:
        output = self.tfm(c, x, src_key_padding_mask=c_padding_mask, 
                          tgt_key_padding_mask=x_padding_mask,
                          memory_key_padding_mask=c_padding_mask) # (bs, seq_len, dim)
        outs = torch.stack([self.residual_decoder[i](output) for i in range(self.n_quantizer)], dim=-1) # (bs, seq_len, logit_dim, n_quant)
        return outs

