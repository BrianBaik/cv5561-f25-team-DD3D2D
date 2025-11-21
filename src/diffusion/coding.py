from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn


# ------------------------------------------------------------------------- #
# Small helpers
# ------------------------------------------------------------------------- #

def _extract(arr_1d: torch.Tensor, t: torch.Tensor, x_shape) -> torch.Tensor:
    """
    Extract values arr_1d[t] for each batch element and reshape for broadcasting.

    arr_1d: (T,) tensor
    t     : (B,) long tensor of indices in [0, T-1]
    x_shape: shape of x, e.g. (B, C, H, W)

    Returns:
        (B, 1, 1, 1, ...) tensor broadcastable over x.
    """
    out = arr_1d.gather(0, t)  # (B,)
    while out.dim() < len(x_shape):
        out = out.unsqueeze(-1)
    return out


def gaussian_kl(
    mu_q: torch.Tensor,
    var_q,
    mu_p: torch.Tensor,
    var_p: torch.Tensor,
) -> torch.Tensor:
    """
    KL( N(mu_q, var_q I) || N(mu_p, var_p I) ) per example.

    Args:
        mu_q: (B, D)
        var_q: scalar or tensor broadcastable to mu_q
        mu_p: (B, D)
        var_p: (B, D) or (B, 1)

    Returns:
        kl_per_example: (B,) KL in *nats* (sum over dimensions).
    """
    B, D = mu_q.shape

    # Make var_q tensor with same shape as mu_q
    if not torch.is_tensor(var_q):
        var_q = torch.tensor(var_q, device=mu_q.device, dtype=mu_q.dtype)
    if var_q.dim() == 0:
        var_q = var_q.view(1).expand(B * D).view(B, D)
    else:
        var_q = var_q.view(B, -1)
        if var_q.shape[1] == 1:
            var_q = var_q.expand(B, D)

    # Ensure var_p has same shape as mu_p
    var_p = var_p.view(B, -1)
    if var_p.shape[1] == 1:
        var_p = var_p.expand(B, D)

    # KL formula
    log_ratio = torch.log(var_p) - torch.log(var_q)
    frac_term = var_q / var_p
    sq_term = (mu_q - mu_p) ** 2 / var_p

    kl = 0.5 * (log_ratio + frac_term + sq_term - 1.0)  # (B, D)
    return kl.sum(dim=1)  # (B,)


# ------------------------------------------------------------------------- #
# Results container
# ------------------------------------------------------------------------- #

@dataclass
class CodingResults:
    xs: List[torch.Tensor]          # xs[t] = x_t for t=0..T, length T+1
    kls_per_step: torch.Tensor      # (B, T+1) KLs in nats [KL_T, KL_{T-1}, ..., KL_0]
    bits_per_dim: torch.Tensor      # (B,) total rate in bits / dim


# ------------------------------------------------------------------------- #
# DiffusionCoder (Algorithm 3 simulator)
# ------------------------------------------------------------------------- #

class DiffusionCoder:
    """
    Simulates Algorithm 3 (sending x_0) for a DDPM.

    Usage:
        schedules = build_schedules(T, ...)
        model = UNet(...)

        coder = DiffusionCoder(model, schedules, device="cuda")
        results = coder.send_x0(x0_batch)

        xs = results.xs
        bits_per_dim = results.bits_per_dim
    """

    def __init__(self, model: nn.Module, schedules: Dict[str, np.ndarray], device: Optional[str] = None):
        self.model = model
        if device is None:
            device = next(model.parameters()).device
        self.device = device

        # Convert schedule arrays (numpy) to torch tensors on this device
        def to_tensor(key):
            return torch.tensor(schedules[key], dtype=torch.float32, device=device)

        self.betas = to_tensor("betas")                                 # (T,)
        self.alphas = to_tensor("alphas")                               # (T,)
        self.alphas_cumprod = to_tensor("alphas_cumprod")               # (T,)
        self.sqrt_alphas_cumprod = to_tensor("sqrt_alphas_cumprod")     # (T,)
        self.sqrt_one_minus_alphas_cumprod = to_tensor("sqrt_one_minus_alphas_cumprod")  # (T,)
        self.posterior_variance = to_tensor("posterior_variance")       # (T,)
        self.posterior_log_variance = to_tensor("posterior_log_variance_clipped")        # (T,)
        self.posterior_mean_coef1 = to_tensor("posterior_mean_coef1")   # (T,)
        self.posterior_mean_coef2 = to_tensor("posterior_mean_coef2")   # (T,)
        self.sigmas = to_tensor("sigmas")                               # (T,)

        self.num_timesteps = self.betas.shape[0]

    # ------------------------------------------------------------------ #
    # Forward q(x_t | x_0) sampling (Eq. 4)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Sample x_t ~ q(x_t | x_0) using:
            x_t = sqrt(ᾱ_t) x_0 + sqrt(1 - ᾱ_t) ε, ε ~ N(0, I)

        Args:
            x0: (B, C, H, W)
            t : (B,) timestep indices in [0, T-1]

        Returns:
            x_t: (B, C, H, W)
        """
        noise = torch.randn_like(x0)
        sqrt_alpha_bar_t = _extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alpha_bar_t = _extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise

    # ------------------------------------------------------------------ #
    # Posterior q(x_{t-1} | x_t, x_0) sampling (Eqs. 6–7)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def q_posterior_sample(
        self,
        x_t: torch.Tensor,
        x0: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample x_{t-1} ~ q(x_{t-1} | x_t, x_0).

        Uses precomputed posterior_mean_coef1/2 and posterior_variance.
        Args:
            x_t: (B, C, H, W)
            x0: (B, C, H, W)
            t  : (B,) timestep indices in [1, T-1]

        Returns:
            x_prev: (B, C, H, W) sampled x_{t-1}
            mu_q  : (B, C, H, W) mean of q
        """
        coef1 = _extract(self.posterior_mean_coef1, t, x_t.shape)
        coef2 = _extract(self.posterior_mean_coef2, t, x_t.shape)
        var = _extract(self.posterior_variance, t, x_t.shape)

        mu_q = coef1 * x0 + coef2 * x_t
        noise = torch.randn_like(x_t)
        x_prev = mu_q + torch.sqrt(var) * noise
        return x_prev, mu_q

    # ------------------------------------------------------------------ #
    # Mean/sigma of p_theta(x_{t-1} | x_t) (same as in sampling.py)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def p_mean_sigma(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True,
        sigma_source: str = "sigmas",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute mean and sigma of p_θ(x_{t-1} | x_t) assuming ε-parameterization.

        Steps:
          1. eps_theta = model(x_t, t)
          2. x0_hat = (x_t - sqrt(1-ᾱ_t) eps_theta) / sqrt(ᾱ_t)
          3. μ_θ(x_t, t) = coef1 * x0_hat + coef2 * x_t
          4. σ_t from schedules (sigmas) or posterior variance.

        Returns:
            mean    : (B, C, H, W)
            sigma_t : (B, 1, 1, 1)
            x0_hat  : (B, C, H, W)
        """
        eps_theta = self.model(x_t, t)  # (B, C, H, W)

        sqrt_alpha_bar_t = _extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha_bar_t = _extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )

        # x̂_0
        x0_hat = (x_t - sqrt_one_minus_alpha_bar_t * eps_theta) / sqrt_alpha_bar_t
        if clip_denoised:
            x0_hat = x0_hat.clamp(-1.0, 1.0)

        # μ_θ from posterior coefficients
        coef1 = _extract(self.posterior_mean_coef1, t, x_t.shape)
        coef2 = _extract(self.posterior_mean_coef2, t, x_t.shape)
        mean = coef1 * x0_hat + coef2 * x_t

        if sigma_source == "sigmas":
            sigma_t = _extract(self.sigmas, t, x_t.shape)
        elif sigma_source == "posterior":
            sigma_t = torch.exp(
                0.5 * _extract(self.posterior_log_variance, t, x_t.shape)
            )
        else:
            raise ValueError(f"Unknown sigma_source: {sigma_source}")

        return mean, sigma_t, x0_hat

    # ------------------------------------------------------------------ #
    # Algorithm 3: Sending x_0
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def send_x0(
        self,
        x0: torch.Tensor,
        sigma_source: str = "sigmas",
        clip_denoised: bool = True,
    ) -> CodingResults:
        """
        Simulate Algorithm 3 (sending x_0) and compute rate in bits/dim.

        Args:
            x0           : (B, C, H, W) original images
            sigma_source : which σ_t to use for p_θ (same choices as sampler)
            clip_denoised: clamp x̂_0 to [-1,1] when forming μ_θ

        Returns:
            CodingResults(xs, kls_per_step, bits_per_dim)
        """
        device = self.device
        x0 = x0.to(device)
        B = x0.shape[0]
        T = self.num_timesteps

        # store x_t for t = 0..T
        xs: List[Optional[torch.Tensor]] = [None] * (T + 1)
        xs[0] = x0

        kls: List[torch.Tensor] = []

        # ---------- 1) KL(q(x_T | x_0) || p(x_T)) ----------
        t_T = torch.full((B,), T - 1, device=device, dtype=torch.long)

        # sample x_T using forward process
        x_T = self.q_sample(x0, t_T)   # uses ᾱ_{T-1}
        xs[T] = x_T

        alpha_bar_Tm1 = self.alphas_cumprod[T - 1]
        sqrt_alpha_bar_Tm1 = torch.sqrt(alpha_bar_Tm1)
        var_q_T = 1.0 - alpha_bar_Tm1

        mu_q_T = (sqrt_alpha_bar_Tm1 * x0).view(B, -1)
        mu_p_T = torch.zeros_like(mu_q_T)
        var_p_T = torch.ones_like(mu_q_T)

        kl_T = gaussian_kl(mu_q_T, var_q_T, mu_p_T, var_p_T)  # (B,)
        kls.append(kl_T)

        # ---------- 2) For t = T-1, ..., 1 : KL(q(x_{t-1}|x_t,x0) || p_theta(x_{t-1}|x_t)) ----------
        x_next = x_T  # current x_t in the loop
        for t_idx in range(T - 1, 0, -1):
            t_batch = torch.full((B,), t_idx, device=device, dtype=torch.long)

            # q(x_{t-1} | x_t, x_0)
            x_t, mu_q = self.q_posterior_sample(x_next, x0, t_batch)
            xs[t_idx] = x_t

            # p_theta(x_{t-1} | x_t) ~ N(mean_p, sigma_p^2 I)
            mean_p, sigma_p, _ = self.p_mean_sigma(
                x_next, t_batch, clip_denoised=clip_denoised, sigma_source=sigma_source
            )

            mu_q_flat = mu_q.view(B, -1)
            mu_p_flat = mean_p.view(B, -1)
            var_q_t = self.posterior_variance[t_idx]
            var_p_flat = (sigma_p ** 2).view(B, -1)

            kl_t = gaussian_kl(mu_q_flat, var_q_t, mu_p_flat, var_p_flat)
            kls.append(kl_t)

            x_next = x_t

        # ---------- 3) Final term for sending x_0 via p_theta(x_0 | x_1) ----------
        x1 = xs[1]  # x_1
        t0_batch = torch.zeros(B, device=device, dtype=torch.long)

        # approximate q(x_0 | x_1, x_0) as N(x_0, posterior_variance[0] I)
        mu_q0_flat = x0.view(B, -1)
        var_q0 = self.posterior_variance[0]

        # p_theta(x_0 | x_1)
        mean_p0, sigma_p0, _ = self.p_mean_sigma(
            x1, t0_batch, clip_denoised=clip_denoised, sigma_source=sigma_source
        )
        mu_p0_flat = mean_p0.view(B, -1)
        var_p0_flat = (sigma_p0 ** 2).view(B, -1)

        kl_0 = gaussian_kl(mu_q0_flat, var_q0, mu_p0_flat, var_p0_flat)
        kls.append(kl_0)

        # ---------- 4) Stack KLs and compute bits/dim ----------
        # Order is [KL_T, KL_{T-1}, ..., KL_1, KL_0]
        kls_per_step = torch.stack(kls, dim=1)  # (B, T+1)

        total_kl = kls_per_step.sum(dim=1)      # (B,) total nats over all dims
        num_dims = x0.view(B, -1).shape[1]
        bits_per_dim = total_kl / (num_dims * torch.log(torch.tensor(2.0, device=device)))

        # cast xs[0]–xs[T] to non-Optional for type checkers
        xs = [x for x in xs]  # type: ignore

        return CodingResults(xs=xs, kls_per_step=kls_per_step, bits_per_dim=bits_per_dim)

    # ------------------------------------------------------------------ #
    # Reconstruction from x_t (Eq. 15) – useful for distortion
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def reconstruct_from_xt(self, x_t: torch.Tensor, t_scalar: int) -> torch.Tensor:
        """
        Given x_t at integer timestep t, compute the current prediction of x_0:

            x̂_0 = (x_t - sqrt(1 - ᾱ_t) ε_θ(x_t, t)) / sqrt(ᾱ_t)

        Args:
            x_t    : (B, C, H, W)
            t_scalar: integer in [0, T-1]

        Returns:
            x0_hat: (B, C, H, W) in [-1, 1]
        """
        B = x_t.shape[0]
        t = torch.full((B,), t_scalar, device=self.device, dtype=torch.long)

        eps_theta = self.model(x_t, t)
        sqrt_alpha_bar_t = _extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha_bar_t = _extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )

        x0_hat = (x_t - sqrt_one_minus_alpha_bar_t * eps_theta) / sqrt_alpha_bar_t
        return x0_hat.clamp(-1.0, 1.0)
