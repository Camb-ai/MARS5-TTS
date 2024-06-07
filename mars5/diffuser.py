"""
Discrete multinomial diffusion code adapted from https://github.com/RF5/transfusion-asr,
which in turn is adapted from https://github.com/ehoogeboom/multinomial_diffusion.

Please see the original repo (https://github.com/ehoogeboom/multinomial_diffusion) and paper for full
details on how multinomial diffusion works -- thanks to the original authors!
"""

import torch
from torch import Tensor
from torch.functional import F
import numpy as np
from dataclasses import dataclass
from typing import Union

# -------------- Multinomial utility functions -----------

MIN_LOG_ARG = 1e-7 # originally was 1e-40

def log_1_min_a(a): return torch.log((1 - a.exp()).clamp_(min=1e-30))

def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

def extract(a: Tensor, t, x_shape):
    """ Given 1D vector of alpha/alpha_cum/betas, get index at `t` of shape (bs,), and then
    broadcast it to number of dims in `x_shape`. 
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def index_to_log_onehot(x, num_classes, dim=-1, dtype=torch.float32):
    """ Convert indices `x` (bs, ...) to approx one-hot log-probs of shape (bs, ..., num_classes) """
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    if dim == 1:
        permute_order = (0, -1) + tuple(range(1, len(x.size())))
        x_onehot = x_onehot.permute(permute_order)
    else: 
        pass

    log_x = torch.log(x_onehot.to(dtype).clamp(min=MIN_LOG_ARG)) # so min(log_x) will be -30

    return log_x

def sum_except_batch(x: Tensor, num_dims=1) -> Tensor:
    '''
    Sums all dimensions except the first.
    Args:
        x: Tensor, shape (batch_size, ...)
        num_dims: int, number of batch dims (default=1)
    Returns:
        x_sum: Tensor, shape (batch_size,)
    '''
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)

# -------------- Multinomial diffusion class -------------

class MultinomialDiffusion():
    def __init__(self, num_classes, timesteps=100, diffusion_s=0.008,
                 loss_type='vb_stochastic', parametrization='x0', 
                 dtype=torch.float32,
                 device='cpu'):
        super(MultinomialDiffusion, self).__init__()
        assert loss_type in ('vb_stochastic',)
        assert parametrization in ('x0', 'direct')

        self.num_classes = num_classes
        self.loss_type = loss_type
        self.num_timesteps = timesteps
        self.parametrization = parametrization

        alphas = self.cosine_beta_schedule(timesteps, diffusion_s)

        alphas = alphas.to(torch.float64)
        log_alpha = alphas.log()
        log_cumprod_alpha = torch.cumsum(log_alpha, dim=-1)

        log_1_min_alpha = log_1_min_a(log_alpha) # = log(betas)

        log_1_min_cumprod_alpha = log_1_min_a(log_cumprod_alpha) # = log(1- \bar{a}) 
        a = log_add_exp(log_alpha, log_1_min_alpha) # log(1-beta + beta) = log(1) = 0

        assert log_add_exp(log_alpha, log_1_min_alpha).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_alpha, log_1_min_cumprod_alpha).abs().sum().item() < 1e-5
        assert (torch.cumsum(log_alpha, dim=-1) - log_cumprod_alpha).abs().sum().item() < 1.e-5

        # Convert to float32 and register buffers.
        self.log_alpha = log_alpha.to(dtype).to(device)
        self.log_1_min_alpha = log_1_min_alpha.to(dtype).to(device)
        self.log_cumprod_alpha = log_cumprod_alpha.to(dtype).to(device)
        self.log_1_min_cumprod_alpha = log_1_min_cumprod_alpha.to(dtype).to(device)

    @staticmethod
    def cosine_beta_schedule(timesteps, s=0.008) -> Tensor:
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672 .
        Returns alpha parameters, NOT Beta
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
        alphas = torch.clamp(alphas, 0.001, 1.0)
        return torch.sqrt(alphas)

    def multinomial_kl(self, log_prob1: Tensor, log_prob2: Tensor, dim=-1) -> Tensor:
        """ Get KL divergence between two categorical distributions specified with `log_prob1` and `log_prob2`.
        Assumed probability dim is `dim` (i.e. log_prob1.exp().sum(dim=`dim`) should be tensor of ones)
        """
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=dim)
        return kl

    def q_pred_one_timestep(self, log_x_t: Tensor, t: Tensor) -> Tensor:
        """ Compute q(x_t | x_{t-1}) = C(x_t | alpha_t * x_{t-1} + (1-alpha_t)/K in the log-domain
        given `log_x_t` as log one-hot encoding of x_t. 
        
        Recall due to symmetry property we can compute
        this value using x_t instead of x_{t-1} (se appendix A of https://arxiv.org/pdf/2102.05379.pdf)
        """
        dt = log_x_t.dtype
        log_alpha_t = extract(self.log_alpha, t, log_x_t.shape).to(dt)
        log_1_min_alpha_t = extract(self.log_1_min_alpha, t, log_x_t.shape).to(dt)

        # alpha_t * E[xt] + (1 - alpha_t) 1 / K
        log_probs = log_add_exp(
            log_x_t + log_alpha_t,
            log_1_min_alpha_t - np.log(self.num_classes)
        )
        return log_probs

    def q_pred_one_timestep_scaled(self, log_x_t: Tensor, t: Tensor, c: int, jump_len: int) -> Tensor:
        """ Compute q(x_t | x_{t-1}) = C(x_t | alpha_t * x_{t-1} + (1-alpha_t)/K in the log-domain
        given `log_x_t` as log one-hot encoding of x_t. 
        
        Recall due to symmetry property we can compute
        this value using x_t instead of x_{t-1} (se appendix A of https://arxiv.org/pdf/2102.05379.pdf)
        """
        dt = log_x_t.dtype
        log_alpha_t = extract(self.log_alpha, t, log_x_t.shape).to(dt)
        log_1_min_alpha_t = extract(self.log_1_min_alpha, t, log_x_t.shape).to(dt)

        # Magic
        xax = torch.arange(0,log_x_t.shape[1],1).to(log_x_t.device)
        aa=log_x_t.shape[1]*(c/jump_len)
        sig = 1/(1+torch.exp(-(xax-aa+20)/8))
        log_alpha_t = (torch.log(1/sig)[None,:,None] + log_alpha_t).clamp(-torch.inf, 0)
        log_1_min_alpha_t = torch.log(sig)[None,:,None] + log_1_min_alpha_t

        # alpha_t * E[xt] + (1 - alpha_t) 1 / K
        log_probs = log_add_exp(
            log_x_t + log_alpha_t,
            log_1_min_alpha_t - np.log(self.num_classes)
        )
        return log_probs

    def q_pred(self, log_x_start: Tensor, t) -> Tensor:
        """ Compute q(x_t | x_0) = C(x_t | bar{alpha}_t * x_0 + (1 - bar{alpha}_t)/K ) in log domain,
        given `log_x_start` of log probs of x_0.
        """
        dt = log_x_start.dtype
        log_cumprod_alpha_t = extract(self.log_cumprod_alpha, t, log_x_start.shape).to(dt)
        log_1_min_cumprod_alpha = extract(self.log_1_min_cumprod_alpha, t, log_x_start.shape).to(dt)
        
        log_probs = log_add_exp(
            log_x_start + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - np.log(self.num_classes)
        )
        
        return log_probs

    def q_posterior(self, log_x_start, log_x_t, t):
        """ Compute  `q(xt-1 | xt, x0) = q(xt | xt-1, x0) * q(xt-1 | x0) / q(xt | x0)`
        where q(xt | xt-1, x0) = q(xt | xt-1).
        """
        # q(xt-1 | xt, x0) = q(xt | xt-1, x0) * q(xt-1 | x0) / q(xt | x0)
        # where q(xt | xt-1, x0) = q(xt | xt-1).

        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_EV_qxtmin_x0 = self.q_pred(log_x_start, t_minus_1) # log( q(x_{t-1} | x_0) )
        # if t == 0, then log( q(x_0 | x_0) ) = log( one_hot(x_0) ), not even random at that point.
        # so, where t == 0 
        num_axes = (1,) * (len(log_x_start.size()) - 1) 
        t_broadcast = t.view(-1, *num_axes) * torch.ones_like(log_x_start) # broadcast to non-batch axes
        log_EV_qxtmin_x0 = torch.where(t_broadcast == 0, log_x_start, log_EV_qxtmin_x0) 
        # where it is zero, replace
        # with log one-hot encoding of x0.

        # Note: _NOT_ x_tmin1, which is how the formula is typically used!!!
        # Not very easy to see why this is true. But it is :)
        # log_EV_qxtmin_x0 ~ q(x_{t-1} | x_0) 
        # q_pred_one_timestep(log_x_t, t) ~ q(x_t | x_{t-1}) (which due to symmetry can be computed using x_t)
        unnormed_logprobs = log_EV_qxtmin_x0 + self.q_pred_one_timestep(log_x_t, t) # numerator of bayes
        
        # approximate denominator with just a normalizing sum.
        log_EV_xtmin_given_xt_given_xstart = \
            unnormed_logprobs \
            - torch.logsumexp(unnormed_logprobs, dim=-1, keepdim=True)
        
        return log_EV_xtmin_given_xt_given_xstart

    def p_pred(self, log_x_t, t, log_x0_pred):
        """ Predict `p(x_{t-1} | x_t)` using `q(xt-1 | xt, hat{x0})`, where `hat{x0}` is given by
        log probabilities from model as `log_x0_pred` (bs, ...., K) and x_t is given by
        `log_x_t` of shape `(bs, ..., K)`
        """
        # log_x_recon = self.predict_start(log_x, t=t) # model itself predicts x_0
        # log_x0_pred
        log_model_pred = self.q_posterior(
            log_x_start=log_x0_pred, log_x_t=log_x_t, t=t)
        return log_model_pred

    def log_sample_categorical(self, logprobs: Tensor, dim=-1) -> Tensor:
        """ Sample from categorical `logprobs` (bs, ..., probs), where position of probs is specified
        by `dim`.

        Returns sampled long indices of shape `(bs, ...)`
        """
        uniform = torch.rand_like(logprobs)
        gumbel_noise = -torch.log( (-torch.log(uniform.clamp_(min=MIN_LOG_ARG)) ).clamp_(min=MIN_LOG_ARG))
        sample = (gumbel_noise + logprobs).argmax(dim=dim)
        return sample

    def q_sample(self, log_x_start, t):
        """ Draw `x_t` ~ q(x_t | x_0) . `log_x_start` is of shape `(bs, ..., K)`, returns result of same shape """
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)
        sample = self.log_sample_categorical(log_EV_qxt_x0)
        # log_sample = index_to_log_onehot(sample, self.num_classes)

        return sample #log_sample

    def compute_Lt(self, log_x_start: Tensor, log_x_t: Tensor, log_x0_pred: Tensor, t, 
                   detach_mean=False, include_kl_prior=True):
        """ Get loss given one-hot log x_0, one-hot log x_t, t, and model prediction `log_x0_pred`.
        Parameters:
            - `log_x_start`: ground-truth input x0, converted to log one-hot (bs, ..., K)
            - `log_x_t`: sampled noisy input at `x_t`, converted to log one-hot (bs, ..., K)
            - `t`: diffusion timestep (bs,)
            - `log_x0_pred`: model prediction of log probabilities of x0, i.e. hat{x0}.
            - `include_kl_prior`: add last two terms to model loss (does not change optimization problem).
        """
        dtype = log_x_start.dtype
        log_true_prob = self.q_posterior(
            log_x_start=log_x_start, log_x_t=log_x_t, t=t)

        log_model_prob = self.p_pred(log_x_t=log_x_t, t=t, log_x0_pred=log_x0_pred)

        if detach_mean:
            log_model_prob = log_model_prob.detach()

        kl = self.multinomial_kl(log_true_prob, log_model_prob)
        kl = sum_except_batch(kl)

        # Add L_0, -log(p(x_0 | x_1))
        decoder_nll = - (log_x_start.exp() * log_model_prob).sum(dim=-1)
        decoder_nll = sum_except_batch(decoder_nll)

        mask = (t == torch.zeros_like(t)).to(dtype)
        loss = mask * decoder_nll + (1. - mask) * kl # only add L0 if t == 0.

        if include_kl_prior:
            pt = torch.ones_like(t, dtype=dtype)
            kl_prior = self.kl_prior(log_x_start)
            loss = (kl) + kl_prior

        return loss

    def kl_prior(self, log_x_start: Tensor) -> Tensor:
        """ This function computes -H_{q}(x_T | x_0)+H_{p}(x_T), which 
        by some math (see wiki for KL div relation to conditional entropy).
        So KL(q(x_T | x_0) || 1/K) = -H_{q}(x_T | x_0)+H_{p}(x_T) for categorical distribution.

        Given `log_x_start` (bs, ..., probs), return KL prior of shape (bs,)
        """
        b = log_x_start.size(0)
        device = log_x_start.device
        ones = torch.ones(b, device=device, dtype=torch.long)

        log_qxT_prob = self.q_pred(log_x_start, t=(self.num_timesteps - 1) * ones) # q(x_T | x_0)
        log_half_prob = -torch.log(self.num_classes * torch.ones_like(log_qxT_prob)) # log(1/K), broadcast to q(x_T|x_0) shape

        kl_prior = self.multinomial_kl(log_qxT_prob, log_half_prob)
        return sum_except_batch(kl_prior)


def index2logit(x: Tensor, vocab_size: int, dtype=torch.float32):
    x = F.one_hot(x, num_classes=vocab_size).to(dtype)
    x = x * (vocab_size/(vocab_size - 1)) - 1/(vocab_size - 1)
    return x


# ------------------------------
# Functions adapted from the full


@dataclass
class DSH():
    # Diffusion Sampling Hyperparameters [DSH] (Section 4)
    jump_len: int = 1 # j in RePaint paper [default 10] (Section 4.1)
    jump_n_sample: int = 1 # r in RePaint paper [default 10] (Section 4.1)
    last_greedy: bool = False # whether to not sample at t=0, but take argmax prediction. [default False]
    x_0_temp: float = 1.0 # reweight temp for model prediction of x0
    guidance_w: float = 1.0 # classifier free guidance weight [default 1.5] (Section 4.3)
    enable_kevin_scaled_inference: bool = True # sequentially progressive diffusion [default True] (Section 4.2)
    T_override: Union[None, int] = None # allow variable transcription sizes during inference (Section 4.4)

    deep_clone: bool = False # whether to do deep clone. 
    q0_override_steps: int = 0 # number of steps that we allow overriding the input quant level 0 inputs.
    progress: bool = False # whether to show progress bar


def get_schedule(t_T, jump_len=10, jump_n_sample=10):
    jumps = {}
    for j in range(0, t_T - jump_len, jump_len):
        jumps[j] = jump_n_sample - 1
    t = t_T
    ts = []
    while t >= 1:
        t = t-1
        ts.append(t)
        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(jump_len):
                t = t + 1
                ts.append(t)
    ts.append(-1)
    return ts


def forward_diffusion(diff: MultinomialDiffusion, dtype, x, t, c=None, dsh=DSH):
    """Simple forward diffusion process p"""
    log_x_t = index_to_log_onehot(x, diff.num_classes, dtype=dtype)
    if c is not None: x = diff.q_pred_one_timestep_scaled(log_x_t, t, c, dsh.jump_len)
    else: x = diff.q_pred_one_timestep(log_x_t, t)
    x = diff.log_sample_categorical(x)
    return x


def reverse_diffusion(diff: MultinomialDiffusion, model, batch, x_known=None, m=None, 
                      last_greedy=False, temperature=1.0, alphas=None, ensemble_size=1, dsh=DSH):
    """Reverse diffusion process q: predict x_{t-1} given x, t, x_known, m. Optionally do not sample model output
    for t=0, but rather use the greedy argmax with `last_greedy`.
    """
    x = batch[4]
    t = batch[-1]
    if x_known is None: x_known = torch.zeros_like(x)
    if m is None: m = torch.zeros_like(x)

    # Equation 8b
    # for b in batch:
        # print(f"{b.shape}: {b}")
    x_0_pred = model(*batch) # (bs, seq_len, logit_dim, n_quant)
    x_0_pred = x_0_pred.permute(0, 1, 3, 2) # (bs, seq_len, n_quant, dim)

    if dsh.guidance_w != 1:
        uncond_x_0_pred = model(*(c.clone() if c is not None else None for c in batch), drop_cond=True)
        uncond_x_0_pred = uncond_x_0_pred.permute(0, 1, 3, 2)
        x_0_pred = dsh.guidance_w*x_0_pred + (1-dsh.guidance_w)*uncond_x_0_pred

    x_0_pred = x_0_pred / temperature
    log_x_0_pred = F.log_softmax(x_0_pred, dim=-1)
    log_x_t = index_to_log_onehot(x, diff.num_classes, dtype=x_0_pred.dtype)

    # print("PRE: ", log_x_t.shape, t.shape, log_x_0_pred.shape)
    log_model_pred = diff.p_pred(log_x_t, t, log_x_0_pred) # p(x_{t-1} | x_{t})
    
    a_t = alphas[t[0]] if alphas is not None else 0
    mat = torch.eye(ensemble_size, device=x.device)*(1-a_t)
    mat += 1/ensemble_size * a_t
    mat = torch.block_diag(*([mat]*(x.shape[0]//ensemble_size)))
    log_model_pred = ( (mat[..., None, None] ).log().to(x.dtype) + log_model_pred[None])
    log_model_pred = torch.logsumexp(log_model_pred, dim=1)
    
    if (t==0).all() and last_greedy: # Do not sample at t=0
        x_tm1_unknown = log_model_pred.argmax(dim=-1)
    else:
        x_tm1_unknown = diff.log_sample_categorical(log_model_pred)
    
    # Equation 8a
    x_known_log = index_to_log_onehot(x_known, diff.num_classes, dtype=x_0_pred.dtype)
    if (t==0).all(): # Do not sample at t=0
        x_tm1_known = x_known
    else:
        x_tm1_known = diff.q_sample(x_known_log, t)
    
    # Equation 8c
    x_tm1 = x_tm1_known * m.long() + x_tm1_unknown * (1 - m.long())
    return x_tm1, x_0_pred



@torch.inference_mode()
def perform_simple_inference(model: torch.nn.Module, batch: tuple, diff: MultinomialDiffusion, T, dtype=torch.float16,
                             retain_quant0: bool = True, dsh=DSH):
    """ If `retain_quant0`, then do not sample quant0 in each forward or reverse diffusion step. """

    # (bs=1, N), (bs, seq_len2, 8), (bs,)
    c_text, c_codes, c_text_lengths, c_codes_lengths, x, x_padding_mask = batch
        
    device = c_text.device
    bs = c_text.shape[0]
    x_quant0 = x[..., 0].clone() # (bs, seq_len) 0th quant level
    x = torch.randint(0, diff.num_classes, x.shape, dtype=x.dtype, device=device)
    # CRITICAL LINE: override quantization level 0 with provided quant0 level.
    x[..., 0] = x_quant0 

    # RePaint paper resample scheduling
    times = get_schedule(T, jump_n_sample=dsh.jump_n_sample, jump_len=dsh.jump_len)

    x_known = torch.zeros_like(x)
    x_known[..., 0] = x[..., 0] # override L0 codes
    m = torch.zeros_like(x).bool()
    # (bs, seq_len, 8)
    m[..., 0] = True

    offset = 0
    if dsh.deep_clone:
        print(f"Note: using deep clone. Assuming input `c_phones` is concatenated prompt and output phones.",
              "Also assuming no padded indices in `c_codes`.")
        prompt = c_codes
        x = torch.cat((prompt, x), dim=1) # (bs=1, sl1 + sl2, 8)
        x_known = torch.cat((prompt, x_known), dim=1)
        x_padding_mask = torch.cat((
            torch.zeros(x_padding_mask.shape[0], c_codes_lengths[0], dtype=torch.bool, device=x_padding_mask.device), 
            x_padding_mask), dim=-1
        )
        # (bs=1, :up to prompt duration, all 8 codebooks) = True/masked.
        m = torch.cat((torch.ones_like(prompt), m), dim=1)
        x_quant0 = torch.cat((prompt[..., 0], x_quant0), dim=-1)
        offset = c_codes_lengths[0]

        print(f"New x: {x.shape} | new x_known: {x_known.shape} . Base prompt: {prompt.shape}. New padding mask: {x_padding_mask.shape} | m shape: {m.shape}")

    c = 0 # sequentially progressive diffusion offset (Section 4.2)

    # ensemble bs (not in paper)
    alphas = torch.linspace(1, 0, T).to(device)

    pb = zip(times[:-1], times[1:])
    if dsh.progress:
        from fastprogress import progress_bar
        pb = progress_bar(pb, total=len(times)-1)

    # See RePaint paper algorithm
    for t_last, t_cur in pb:

        t = torch.ones((bs,), dtype=torch.long, device=x.device) * (t_last)
        if t_cur < t_last:
            if c > dsh.jump_n_sample:
                c = 0
            c += 1/dsh.jump_len

            # Reverse diffusion: q
            cbatch = (c_text, c_codes, c_text_lengths, c_codes_lengths, x, x_padding_mask, t) 
            x, x_0_pred = reverse_diffusion(diff, model, cbatch, x_known, m, temperature=dsh.x_0_temp, alphas=alphas, ensemble_size=1, dsh=dsh)
        else:
            # Forward diffusion: p
            if dsh.enable_kevin_scaled_inference: x = forward_diffusion(diff, dtype, x, t, c=c, dsh=dsh)
            else: x = forward_diffusion(diff, dtype, x, t, c=None, dsh=dsh)

        if retain_quant0 and dsh.q0_override_steps < t_last:
            x[..., 0] = x_quant0

    # crop offset:
    x = x[:, offset:]
    return x
