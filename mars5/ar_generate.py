import torch
import torch.nn.functional as F
import torchaudio
import copy
from torch import Tensor, nn
import logging
from .model import length_to_mask
from .samplers import (apply_typical_p, early_eos_penalty,
                      top_k_top_p_filtering, freq_rep_penalty)
from .nn_future import RotatingBufferCache
from .minbpe.codebook import CodebookTokenizer
from .minbpe.regex import RegexTokenizer


@torch.inference_mode()
def ar_generate(texttok: RegexTokenizer, speechtok: CodebookTokenizer, 
                codeclm: nn.Module, xx: Tensor, ss_gen: Tensor, first_codex_idx: int, 
                max_len: int = 1500, fp16: bool = True, temperature: float = 1.0, topk: int = None,
                top_p=1.0, alpha_frequency=0, alpha_presence=0, penalty_window=100,
                typical_p=1.0, eos_penalty_factor=1.0, eos_penalty_decay=0, n_phones_gen=None, vocode=True,
                beam_width: int = 1, beam_length_penalty=2, use_kv_cache: bool = True) -> tuple[Tensor, Tensor]:
    """ Use the `codeclm` language model to autoregressively generate a completion of `xx` (seq_len), where the first `first_codex_idx`-1
    indices correspond to the input phones. The output generation is limited to at most `max_len` (measured as num latent codes).
    Returns both output first quantizer codes and synthesized audio using `codec`. Use decoding with `beam_width` to keep 
    track of top `beam_width` outcomes, selecting the top one among them. 

    - Optionally vocode if `vocode` (default True).
    - See `InferenceConfig` for other inference docs. 
    """
    assert xx.dim() == 1, "Only batch size of 1 is currently supported."
    assert beam_width == 1, "Only beam size of 1 is currently supported."
    # internally our batch size will be the beam width
    bs = beam_width
    x_inp = xx[None].repeat(bs, 1) # (bs, seq_len)
    ss_gen = ss_gen[None].repeat(bs, 1, 1)
    # We must subtract 1 in the line below so that we match the train-time conditions of having a
    # False padding value for the <bos> token position. This is needed so that we correctly use the
    # _acoustic_ and not the linguistic language embedding for the <bos> token.
    offsets = torch.tensor([first_codex_idx - 1 for _ in range(bs)], dtype=torch.long, device=xx.device)
    valid_logit_idx_start = len(texttok.vocab) # vocab['s2i']['quant0-0000']
    valid_logit_idx_end = len(texttok.vocab) + len(speechtok.vocab) + 1 # vocab['s2i']['quant1-0000']
    # Make mask that is True where we have valid outputs, False otherwise (where we have text outputs). 
    # logit_mask = torch.zeros(n_vocab, dtype=bool, device=x_inp.device)
    # logit_mask[valid_logit_idx_start:valid_logit_idx_end] = True
    # logit_mask[vocab['s2i']['<eos>']] = True
    cum_logprobs = torch.zeros(bs, dtype=torch.float, device=x_inp.device)
    eos_idx = len(texttok.vocab) + speechtok.special_tokens['<|endofspeech|>']
    n_vocab = len(texttok.vocab) + len(speechtok.vocab)

    logging.info(f"Starting beam decoding with beam_width={beam_width}")

    prev_ids = [[] for _ in range(bs)]

    cache = None
    if use_kv_cache:
        # Initialise kv cache
        cache_window = min(codeclm.ar.args.sliding_window, x_inp.shape[-1] + max_len)
        cache = RotatingBufferCache(codeclm.ar.args.n_layers, bs, cache_window, codeclm.ar.args.n_kv_heads, codeclm.ar.args.head_dim)
        cache.to(device=x_inp.device, dtype=torch.float16 if fp16 else torch.float32)

    counter = 0
    while x_inp.shape[-1] < max_len:
        counter += 1
        gen_length = torch.tensor([x_inp.shape[-1] for _ in range(bs)], dtype=torch.long, device=xx.device)
        padding_mask = length_to_mask(gen_length, offsets)
        
        with torch.autocast('cuda', enabled=fp16):
            logits: Tensor = codeclm(x_inp, padding_mask, spk_reference=ss_gen, cache=cache, counter=counter)
        logits = logits.float()

        logits = logits[:, -1] # select last index, now (bs, logit_dim)

        # <---------------------- logit filtering ---------------------->
        filtered_logits = logits.clone()

        # apply repetition penalty before logit mask if any item in the beam has more than 1 prior token.
        if len(prev_ids[0]) > 1: 
            filtered_logits = freq_rep_penalty(filtered_logits, previous=torch.tensor(prev_ids, dtype=torch.long), 
                                             alpha_frequency=alpha_frequency, alpha_presence=alpha_presence, 
                                             penalty_window=penalty_window)

        filtered_logits[..., :valid_logit_idx_start-1] = float('-inf')
        filtered_logits[..., valid_logit_idx_end:] = float('-inf')

        if n_phones_gen is not None:
            # apply eos penalty
            filtered_logits = early_eos_penalty(filtered_logits, len(prev_ids[0]), n_phones_gen, 
                                                eos_penalty_decay, eos_penalty_factor, 
                                                eos_index=eos_idx)

        filtered_logits = filtered_logits / temperature
        filtered_logits = top_k_top_p_filtering(filtered_logits, top_k=topk, top_p=top_p)
        filtered_logits = apply_typical_p(filtered_logits, mass=typical_p)

        # mask out anything that isn't first quantizer output codes
        filtered_logits[..., :valid_logit_idx_start-1] = float('-inf')
        filtered_logits[..., valid_logit_idx_end:] = float('-inf')
        logits = filtered_logits

        # <---------------------- next frame prediction --------------------->

        logprobs = logits.log_softmax(dim=-1)

        # update assignments: if any beam ended in <eos> last step, it MUST also end in <eos> this step.
        # so, below we multiply the logits with a True/False mask, setting to 
        for j in range(bs):
            if x_inp[j, -1] == eos_idx:
                # do not add any additional probability to it, keeping it the same for all vocab idxs
                logprobs[j] = float('-inf') # zero probability of anything non-eos after 1 eos
                logprobs[j, eos_idx] = 0 # probability=1 of <eos> after <eos>

        candidate_cum_logprobs = cum_logprobs[:, None] + logprobs # (bs, 1) + (bs, vocab) -> (bs, vocab)

        logp_flat = logprobs.flatten()
        candidates = torch.multinomial(logp_flat.exp(), num_samples=beam_width, replacement=False) # (bs,)
        # Ravel it up:
        beam_idxs = candidates // n_vocab # (bs,)
        tok_inds_in_each_beam = candidates % n_vocab # (bs,)                

        # check for breaks
        if torch.all(tok_inds_in_each_beam == eos_idx):
            # apply length penalty:
            non_eos_toks = (x_inp != eos_idx).sum(dim=-1) # (bs,) number of non eos toks
            gen_length = non_eos_toks - first_codex_idx
            penalties = (gen_length**beam_length_penalty)
            penalized_cum_tok_logp = candidate_cum_logprobs / penalties[:, None] 

            eos_avg_logps = penalized_cum_tok_logp[:, eos_idx]
            best_beam_idx = eos_avg_logps.argmax()
            best_avg_logp = eos_avg_logps[best_beam_idx]
            best_beam = x_inp[best_beam_idx]
            logging.info((f"best beam = {best_beam_idx} @ penalized_cum_tok_logp = {best_avg_logp.item():.3f} |\n num toks: {non_eos_toks.cpu().tolist()}. "
                         f"Candidates: {eos_avg_logps.cpu()} |\n non-eos toks: {non_eos_toks.cpu().tolist()} |\n penalties: {penalties.cpu().tolist()} | "
                         f"raw cumulative probs: {candidate_cum_logprobs[:, eos_idx].cpu().tolist()}"))
            break

        # update beam histories:
        x_inp = x_inp[beam_idxs]
        # update next token
        next_sample = tok_inds_in_each_beam
        # update cum logprob
        cum_logprobs = cum_logprobs[beam_idxs] + logprobs[beam_idxs, tok_inds_in_each_beam]
        # update prior inds to point to correct beam
        prev_ids = [copy.deepcopy(prev_ids[beam_idx.item()]) for beam_idx in beam_idxs]
        # add new tokens to previous ids
        for j in range(bs):
            prev_ids[j].append(tok_inds_in_each_beam[j].item())

        logging.debug("L%d | next sample: %s | beam: %s | cum_logp: %s", len(x_inp[0]), next_sample.cpu().tolist(), beam_idxs.cpu().tolist(), cum_logprobs.cpu())

        # update cache with beam indexes
        if cache is not None:
            cache.cache_k = cache.cache_k[:, beam_idxs]
            cache.cache_v = cache.cache_v[:, beam_idxs]
        
        # add 1 None below to make (bs,) -> (bs, 1) so we can concat along seq len dim.
        x_inp = torch.cat([x_inp, next_sample[:, None]], dim=-1)
        

    if x_inp.shape[-1] >= max_len - 1:
        logging.warning(f"[autoregressive generation] output length = {x_inp.shape[-1]} -- inference likely failed or input too long!")
        best_beam = x_inp[0]

    if not vocode: return best_beam # (seq_len,)
    else: raise AssertionError()
