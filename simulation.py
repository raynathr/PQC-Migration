"""
simulation.py

Core Monte Carlo engine for the PQC migration risk model. All
computation is vectorized across a (N_iterations, T_years) grid using
NumPy -- no Python-level loops over iterations or years in the hot path.

Equation references correspond to the paper draft (main.tex):
  - capability-growth : log10(A(t)) = log10(A0) + g*t + eps_t
  - gap               : delta(t) = log10(A(t)) - C_alg
  - rqr               : RQR(t) = 1 / (1 + exp(-alpha * delta(t)))
  - cas               : CAS(t) = w_AS*M_AS + w_KM*M_KM + w_DC*M_DC + w_CAI*M_CAI
  - tci               : TCI = mean_t(CAS(t))
"""

import numpy as np


def sample_growth_rates(rng: np.random.Generator, n_iterations: int,
                         cfg) -> np.ndarray:
    """
    Sample g, ONCE per Monte Carlo iteration, from the growth-rate
    distribution selected by cfg.GROWTH_MODEL (paper Sec. growth).

    g represents the long-run trend for a given simulated "world" /
    iteration; it is deliberately NOT resampled per year. Resampling it
    annually would collapse the model's distinction between trend
    uncertainty (g) and year-to-year noise around that trend (eps_t),
    which are conceptually different sources of uncertainty in the
    paper's formulation.

    "gamma": g ~ Gamma(k, theta) -- the paper's calibrated default.
        Strictly positive by construction (Pr(g<0)=0), which is why
        the paper adopts it over the Gaussian fit below.
    "gaussian": g ~ Normal(mu_g, sigma_g^2) -- retained ONLY to
        reproduce the paper's documented finding that the Gaussian fit
        to the same two calibration targets implies Pr(g<0)~=0.478,
        an unphysical result. Not used as the simulation default.

    Returns shape (n_iterations,).
    """
    if cfg.GROWTH_MODEL == "gamma":
        return rng.gamma(shape=cfg.GAMMA_K, scale=cfg.GAMMA_THETA, size=n_iterations)
    elif cfg.GROWTH_MODEL == "gaussian":
        return rng.normal(loc=cfg.MU_G, scale=cfg.SIGMA_G, size=n_iterations)
    else:
        raise ValueError(f"Unknown GROWTH_MODEL '{cfg.GROWTH_MODEL}'")


def sample_annual_noise(rng: np.random.Generator, n_iterations: int,
                         t_years: int, sigma_eps: float) -> np.ndarray:
    """
    Sample eps_t ~ Normal(0, sigma_eps^2) independently for EACH
    iteration and EACH year.

    Returns shape (n_iterations, t_years).
    """
    return rng.normal(loc=0.0, scale=sigma_eps, size=(n_iterations, t_years))


def log_capability(A0: float, g: np.ndarray, eps: np.ndarray,
                    years: np.ndarray) -> np.ndarray:
    """
    Implements Eq. capability-growth:
        log10(A(t)) = log10(A0) + g*t + eps_t

    Parameters
    ----------
    A0 : scalar, initial adversarial capability (operations)
    g : array, shape (n_iterations,) -- growth rate per iteration
    eps : array, shape (n_iterations, t_years) -- annual noise
    years : array, shape (t_years,) -- the year index t = 1..T

    Returns
    -------
    array, shape (n_iterations, t_years) -- log10(A(t)) for every
    (iteration, year) pair.
    """
    g_col = g[:, np.newaxis]              # (n_iterations, 1) for broadcasting
    trend = g_col * years[np.newaxis, :]  # (n_iterations, t_years)
    return np.log10(A0) + trend + eps


def capability_gap(log_A: np.ndarray, C_alg: float) -> np.ndarray:
    """
    Implements Eq. gap:
        delta(t) = log10(A(t)) - C_alg

    C_alg must already be in the SAME computational model (classical or
    quantum) as A(t) -- see config.py's unit-consistency note. This
    function does not check that for you; config.get_cost() is where
    that safety check lives.
    """
    return log_A - C_alg


def residual_quantum_risk(delta: np.ndarray, alpha: float) -> np.ndarray:
    """
    Implements Eq. rqr:
        RQR(t) = 1 / (1 + exp(-alpha * delta(t)))

    Clipped internally to avoid overflow in exp() for large |delta|;
    this does not change the mathematical result, it only prevents
    NumPy overflow warnings for extreme (and not physically meaningful)
    capability gaps.
    """
    exponent = np.clip(-alpha * delta, -700, 700)  # np.exp overflows beyond ~709
    return 1.0 / (1.0 + np.exp(exponent))


def composite_assurance_score(rqr: np.ndarray, L_t: np.ndarray,
                               cfg) -> np.ndarray:
    """
    Implements Eq. cas:
        CAS(t) = w_AS*M_AS(t) + w_KM*M_KM(t) + w_DC*M_DC(t) + w_CAI*M_CAI(t)

    M_AS(t) = 1 - RQR(t)
    M_KM(t) = cfg.M_KM_CONSTANT   (PLACEHOLDER -- see config.py)
    M_DC(t) = L_t                 (deployment coverage)
    M_CAI(t) = cfg.M_CAI_CONSTANT (PLACEHOLDER -- see config.py)

    rqr and L_t must be broadcastable to the same shape (typically both
    (n_iterations, t_years), with L_t often identical across iterations
    since deployment coverage in this version of the model is
    deterministic given a scenario, not stochastic).
    """
    M_AS = 1.0 - rqr
    M_KM = cfg.M_KM_CONSTANT
    M_DC = L_t
    M_CAI = cfg.M_CAI_CONSTANT

    return (cfg.W_AS * M_AS) + (cfg.W_KM * M_KM) + (cfg.W_DC * M_DC) + (cfg.W_CAI * M_CAI)


def trust_continuity_index(cas: np.ndarray) -> np.ndarray:
    """
    Implements Eq. tci:
        TCI = (1/T) * sum_t(CAS(t))

    Averages over the year axis (axis=-1), leaving one TCI value per
    Monte Carlo iteration so the caller can report both mean and
    standard deviation across iterations, rather than a single point
    estimate.

    Input shape (n_iterations, t_years) -> output shape (n_iterations,)
    """
    return np.mean(cas, axis=-1)


def run_rqr_simulation(cfg, C_alg: float, rng: np.random.Generator) -> dict:
    """
    Run the full Monte Carlo simulation for RQR evolution for a single
    algorithm / cost combination.

    Returns a dict with:
      - 'log_A'  : (n_iterations, t_years) log10 capability
      - 'delta'  : (n_iterations, t_years) capability gap
      - 'rqr'    : (n_iterations, t_years) residual quantum risk
      - 'years'  : (t_years,) year indices used
    """
    years = np.arange(1, cfg.T_YEARS + 1)

    g = sample_growth_rates(rng, cfg.N_ITERATIONS, cfg)
    eps = sample_annual_noise(rng, cfg.N_ITERATIONS, cfg.T_YEARS, cfg.SIGMA_EPS)

    log_A = log_capability(cfg.A0, g, eps, years)
    delta = capability_gap(log_A, C_alg)
    rqr = residual_quantum_risk(delta, cfg.ALPHA)

    return {"log_A": log_A, "delta": delta, "rqr": rqr, "years": years}


def run_cas_simulation(cfg, C_alg: float, L_t: np.ndarray,
                        rng: np.random.Generator) -> dict:
    """
    Run the full Monte Carlo simulation for CAS/TCI under a given
    deployment scenario L(t).

    L_t must already be evaluated at the same years used for the RQR
    simulation (shape (t_years,)); it is broadcast across all
    iterations since deployment coverage is currently modeled as
    deterministic given a scenario choice (see scenarios.py). This is
    itself a simplifying assumption -- a future version could treat
    migration progress as stochastic too.

    Returns a dict with:
      - 'rqr' : (n_iterations, t_years)
      - 'cas' : (n_iterations, t_years)
      - 'tci' : (n_iterations,)
      - 'years' : (t_years,)
    """
    rqr_result = run_rqr_simulation(cfg, C_alg, rng)
    rqr = rqr_result["rqr"]
    years = rqr_result["years"]

    L_broadcast = L_t[np.newaxis, :]  # (1, t_years) -> broadcasts against (n_iter, t_years)
    cas = composite_assurance_score(rqr, L_broadcast, cfg)
    tci = trust_continuity_index(cas)

    return {"rqr": rqr, "cas": cas, "tci": tci, "years": years}


def scan_parameter(*args, **kwargs):
    """
    Placeholder for the sensitivity-analysis module described in the
    paper (Sec. sensitivity, Fig. 3-equivalent heatmap). Not implemented
    in this pass -- see paper Section XIII for the parameters this will
    need to sweep (mu_g, w_AS, alpha, hybrid overhead).
    """
    raise NotImplementedError(
        "Sensitivity analysis module -- see paper Section XIII. "
        "Not built in this pass; core engine must be validated first."
    )
