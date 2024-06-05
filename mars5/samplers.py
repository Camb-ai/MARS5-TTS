""" 
Code for modifying categorical distributions to improve quality of sampling. 

Adapted from:
- https://github.com/e-c-k-e-r/vall-e/blob/master/vall_e/samplers.py 
- Mirosoft UniLM
- Matthew Baas's typical sampling code. 
- https://github.com/LostRuins/koboldcpp
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
import logging

from torch import Tensor, nn


def freq_rep_penalty(logits: Tensor, previous: Tensor, alpha_frequency: float, alpha_presence: float, penalty_window: int = 100) -> Tensor:
    """ Apply frequency and presence penalty according to openai's formuation.
    Concretely: given `logits` (bs, vocab_size) and `previous` (bs, seq_len,)

    Modified to support batched inference.
    
    See: https://platform.openai.com/docs/guides/text-generation/parameter-details
    """
    bs = logits.shape[0]
    previous = previous[..., -penalty_window:]
    c = torch.zeros_like(logits, device=logits.device, dtype=torch.long) # (1, vocab_size)
    for i in range(bs):
        vals, cnts = previous[i].unique(return_counts=True)
        c[i, vals] = cnts.to(c.device)
    
    logits = logits - c * alpha_frequency - (c > 0).to(logits.dtype) * alpha_presence
    return logits


def early_eos_penalty(logits: Tensor, n_generated: int, estimated_gen_length: int, decay: float, factor: float = 1, eos_index: int = 0) -> Tensor:
    """ Penalize the `eos_index` of `logits` (bs, vocab_size) up to `estimated_gen_length`, 
    whereby we reduce the logit value by `factor`*(expected_length - current_length)^decay,
    `n_generated` is the current number of generated samples. `decay` anneals the penalty relative to the distance.

    Good values for decay are between 0 and 1. 0 = hard always apply penalty of 1, 1 = linearly scale penalty relative to distance. 
    Setting factor = 0 disabled penatly. Increasing factor increases penalty. 
    """
    if n_generated > estimated_gen_length: return logits
    penalty = max(estimated_gen_length - n_generated, 1)

    bigger = logits[:, eos_index] > 0

    modifier = factor*(penalty ** decay) 
    # logits[bigger, eos_index] /= modifier
    # logits[~bigger, eos_index] *= modifier
    logits[:, eos_index] -= modifier
    return logits


# Credit to https://github.com/microsoft/unilm/blob/master/xtune/src/transformers/modeling_utils.py#L1145 /
#  https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
def top_k_top_p_filtering( logits: Tensor, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens=1 ) -> Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens per batch example in the output
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens > 1:
            # Keep at least min_tokens (set to min_tokens-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

    return logits


def apply_typical_p(logprobs: Tensor, mass: float) -> Tensor:
    """ Warp categorical logprobs associated with `x` to be in line with `mass`. Last dimension is the bin dimension. 
    `mass` corresponds to `tau` in the paper. 
    """
    if mass > 0.999: return logprobs
    # see: https://arxiv.org/abs/2202.00666
    # calculate entropy
    # normalized = logprobs #torch.nn.functional.log_softmax(scores, dim=-1)
    normalized = torch.nn.functional.log_softmax(logprobs, dim=-1)
    p = torch.exp(normalized)
    ent = -(normalized * p).nansum(-1, keepdim=True)

    # shift and sort
    shifted_scores = torch.abs((-normalized) - ent)
    sorted_scores, sorted_indices = torch.sort(shifted_scores, descending=False)
    sorted_logits = logprobs.gather(-1, sorted_indices)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

    # Remove tokens with cumulative mass above the threshold
    last_ind = (cumulative_probs < mass).sum(dim=1)
    last_ind[last_ind < 0] = 0
    sorted_indices_to_remove = sorted_scores > sorted_scores.gather(1, last_ind.view(-1, 1))

    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

    scores = logprobs.masked_fill(indices_to_remove, -float('Inf'))
    return scores