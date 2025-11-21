import numpy as np

def get_beta_schedule(schedule_name, num_diffusion_timesteps, beta_start=1e-4, beta_end=0.02):
    """
    Generate a beta schedule for diffusion processes.

    Parameters:
    - schedule_name (str): The type of schedule to generate. Options are 'linear', 'quadratic', 'cosine'.
    - num_diffusion_timesteps (int): The number of timesteps in the diffusion process.
    - beta_start (float): The starting value of beta for linear and quadratic schedules.
    - beta_end (float): The ending value of beta for linear and quadratic schedules.

    Returns:
    - np.ndarray: An array of beta values for each timestep.
    """
    if schedule_name == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == "quadratic":
        betas = np.linspace(np.sqrt(beta_start), np.sqrt(beta_end), num_diffusion_timesteps, dtype=np.float64) ** 2
    elif schedule_name == "cosine":
        timesteps = np.arange(num_diffusion_timesteps + 1, dtype=np.float64) / num_diffusion_timesteps
        alphas_cumprod = np.cos((timesteps + 0.008) / 1.008 * np.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = np.clip(betas, 0, 0.999)
    else:
        raise ValueError(f"Unknown beta schedule: {schedule_name}")

    return betas

def compute_alphas(betas):
    """
    Compute alpha values and their cumulative products from beta values.

    Parameters:
    - betas (np.ndarray): An array of beta values.

    Returns:
    - tuple: A tuple containing:
        - alphas (np.ndarray): An array of alpha values.
        - alphas_cumprod (np.ndarray): An array of cumulative products of alpha values.
    """
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    return alphas, alphas_cumprod

def build_schedules(num_diffusion_timesteps, schedule_name="linear", beta_start=1e-4, beta_end=0.02, sigma_type="beta"):
    """
    Build diffusion noise and posterior schedules for a given number of timesteps.

    This computes the forward-process betas and derived quantities used in DDPM-style
    diffusion models, including alphas, cumulative products, square roots, and
    posterior mean/variance terms. It also selects a per-timestep sigma schedule
    based on the requested `sigma_type`.

    Args:
        num_diffusion_timesteps (int):
            Total number of diffusion timesteps T.
        schedule_name (str, optional):
            Name of the beta schedule to use. Passed directly to
            `get_beta_schedule`. Common values include "linear" and "cosine".
            Defaults to "linear".
        beta_start (float, optional):
            Starting value of beta for the schedule (for schedules that use it).
            Defaults to 1e-4.
        beta_end (float, optional):
            Ending value of beta for the schedule (for schedules that use it).
            Defaults to 0.02.
        sigma_type (str, optional):
            How to define the per-timestep sigmas:
                - "beta":      sigma_t = sqrt(beta_t)
                - "posterior": sigma_t = sqrt(posterior_variance_t)
            Defaults to "beta".

    Returns:
        dict[str, np.ndarray]:
            A dictionary containing the following 1D arrays of length
            `num_diffusion_timesteps`:

                - "betas": betas_t
                - "alphas": alphas_t = 1 - betas_t
                - "alphas_cumprod": prod_{s <= t} alphas_s
                - "alphas_cumprod_prev": prod_{s < t} alphas_s, with 1.0 prepended
                - "sqrt_alphas_cumprod": sqrt(alphas_cumprod)
                - "sqrt_one_minus_alphas_cumprod": sqrt(1 - alphas_cumprod)
                - "sqrt_recip_alphas": sqrt(1 / alphas)
                - "sqrt_recipm1_alphas": sqrt(1 / alphas - 1)
                - "posterior_variance": Var(q(x_{t-1} | x_t, x_0))
                - "posterior_log_variance_clipped": log(posterior_variance) with
                  a small epsilon floor for numerical stability
                - "posterior_mean_coef1": coefficient on x_0 in the posterior mean
                - "posterior_mean_coef2": coefficient on x_t in the posterior mean
                - "sigmas": per-timestep sigmas as chosen by `sigma_type`

    Raises:
        ValueError: If `sigma_type` is not one of {"beta", "posterior"}.
    """
    betas = get_beta_schedule(schedule_name, num_diffusion_timesteps, beta_start, beta_end)
    alphas, alphas_cumprod = compute_alphas(betas)
    
    alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
    sqrt_recip_alphas = np.sqrt(1.0 / alphas)
    sqrt_recipm1_alphas = np.sqrt(1.0 / alphas - 1)
    
    # posterior variance and its log
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    posterior_log_variance_clipped = np.log(np.maximum(posterior_variance, 1e-20))
    
    # posterior mean coefficients
    posterior_mean_coef1 = (betas * np.sqrt(alphas_cumprod_prev)) / (1.0 - alphas_cumprod)
    posterior_mean_coef2 = ((1.0 - alphas_cumprod_prev) * np.sqrt(alphas)) / (1.0 - alphas_cumprod)
    
    if sigma_type == "beta":
        sigmas = np.sqrt(betas)
    elif sigma_type == "posterior":
        sigmas = np.sqrt(posterior_variance)
    else:
        raise ValueError(f"Unknown sigma type: {sigma_type}")
    
    schedules = {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "sqrt_recip_alphas": sqrt_recip_alphas,
        "sqrt_recipm1_alphas": sqrt_recipm1_alphas,
        "posterior_variance": posterior_variance,
        "posterior_log_variance_clipped": posterior_log_variance_clipped,
        "posterior_mean_coef1": posterior_mean_coef1,
        "posterior_mean_coef2": posterior_mean_coef2,
        "sigmas": sigmas,
    }   
    
    return schedules