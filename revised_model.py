"""
revised_model.py

The revised end-to-end pipeline: normalized (A0-free) capability growth
-> probabilistic attack cost -> per-algorithm RQR -> exposure-adjusted
organizational RQR -> operationalized CAS -> TCI.

Normalized coordinates (see calibration.py for why A0 drops out):

    ghat  ~ Gamma(khat, thetahat)      fraction of the RSA-2048 capability
                                        deficit closed per year
    d_alg = (C_alg - log10 A0) / (C_RSA - log10 A0)   relative deficit
    Delta_hat_alg(t) = ghat * t - d_alg + eps_hat_t
    RQR_alg(t) = sigmoid(alpha_tilde * Delta_hat_alg(t))

d_RSA = 1 identically, so RSA-2048's trajectory carries no dependence on
A0 at all. Every other algorithm's d_alg retains a residual dependence,
which is reported as a sensitivity rather than hidden in a constant.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

import costs
import exposure
import metrics

# --- calibrated normalized growth distribution (calibration.py) -------
KHAT = 0.6748867936724054
THETAHAT = 0.11403505557339036
ALPHA_TILDE = 9.62          # = alpha(2.0) * D_RSA(4.81)
SIGMA_EPS_HAT = 0.05 / 4.81  # annual noise, expressed in normalized units

LOG10_A0_NOMINAL = 5.0       # nuisance parameter; swept, never reported alone
LOG10_A0_BRACKET = (2.0, 7.0)


@dataclass
class RevisedConfig:
    N: int = 5000
    T: int = 15
    seed: int = 42
    khat: float = KHAT
    thetahat: float = THETAHAT
    alpha_tilde: float = ALPHA_TILDE
    sigma_eps_hat: float = SIGMA_EPS_HAT
    log10_A0: float = LOG10_A0_NOMINAL
    stochastic_cost: bool = True
    cost_jitter: float = 0.0
    w_AS: float = 0.40
    w_KM: float = 0.25
    w_DC: float = 0.20
    w_CAI: float = 0.15
    org_profile: str = "typical"      # "typical" | "mature"
    hybrid_overhead: float = metrics.HYBRID_OVERHEAD_DEFAULT
    overhead_tolerance: float = metrics.OVERHEAD_TOLERANCE
    C_rsa_ref: float = 9.81


def relative_deficit(cfg: RevisedConfig, C_alg) -> np.ndarray:
    return (np.asarray(C_alg) - cfg.log10_A0) / (cfg.C_rsa_ref - cfg.log10_A0)


def sample_ghat(cfg, rng, n=None):
    return rng.gamma(cfg.khat, cfg.thetahat, size=(n or cfg.N))


def rqr_paths(cfg, alg, rng, ghat=None, eps=None):
    """
    RQR_alg over the horizon, shape (N, T).

    When cfg.stochastic_cost is True, C_alg is drawn per iteration from
    its published-estimate mixture (costs.py), so between-estimate
    disagreement propagates into the output distribution instead of
    being resolved by fiat before the simulation starts.
    """
    years = np.arange(1, cfg.T + 1, dtype=float)
    if ghat is None:
        ghat = sample_ghat(cfg, rng)
    if eps is None:
        eps = rng.normal(0.0, cfg.sigma_eps_hat, size=(len(ghat), cfg.T))

    if cfg.stochastic_cost:
        C = costs.sample_cost(alg, rng, size=len(ghat), jitter=cfg.cost_jitter)
    else:
        C = np.full(len(ghat), costs.POINT_COSTS[alg], dtype=float)

    d = relative_deficit(cfg, C)[:, None]
    delta = ghat[:, None] * years[None, :] - d + eps
    return 1.0 / (1.0 + np.exp(-np.clip(cfg.alpha_tilde * delta, -700, 700)))


ALL_ALGS = ["RSA-2048", "ECC-P256", "Kyber-512", "Kyber-768", "Kyber-1024"]


def all_rqr(cfg, rng_seed=None, algs=ALL_ALGS):
    """
    RQR paths for every algorithm, sharing ONE draw of ghat and eps
    across algorithms. Sharing the capability draw is what makes the
    asset-weighted mixture in exposure.org_rqr coherent: within a single
    simulated future, the same adversary faces every algorithm.
    """
    rng = np.random.default_rng(cfg.seed if rng_seed is None else rng_seed)
    ghat = sample_ghat(cfg, rng)
    eps = rng.normal(0.0, cfg.sigma_eps_hat, size=(cfg.N, cfg.T))
    return {a: rqr_paths(cfg, a, rng, ghat=ghat, eps=eps) for a in algs}, ghat


def cas_tci(cfg, base_L, classes=None, rqr_by_alg=None, rng_seed=None):
    """
    Full CAS / TCI evaluation under a migration schedule base_L.

    Returns a dict with the organizational RQR paths, CAS paths, TCI
    samples, and the deterministic component trajectories.
    """
    classes = classes or exposure.ORG_X
    t = np.arange(1, cfg.T + 1, dtype=float)

    if rqr_by_alg is None:
        rqr_by_alg, _ = all_rqr(cfg, rng_seed=rng_seed)

    rqr_org = exposure.org_rqr(rqr_by_alg, base_L, classes, t)   # (N, T)
    L_eff = exposure.effective_coverage(base_L, classes, t)      # (T,)

    mkm = metrics.m_km(L_eff, cfg.org_profile)
    mcai = metrics.m_cai(L_eff, cfg.org_profile,
                          hybrid_overhead=cfg.hybrid_overhead,
                          overhead_tolerance=cfg.overhead_tolerance)

    cas = (cfg.w_AS * (1.0 - rqr_org)
           + cfg.w_KM * mkm[None, :]
           + cfg.w_DC * L_eff[None, :]
           + cfg.w_CAI * mcai[None, :])
    tci = cas.mean(axis=1)
    return {"rqr_org": rqr_org, "cas": cas, "tci": tci,
            "L_eff": L_eff, "M_KM": mkm, "M_CAI": mcai, "years": t}


# --- migration schedules ---------------------------------------------
def aggressive(t, L_max=1.0, k=1.2, t0=3.0):
    return np.minimum(1.0, L_max / (1.0 + np.exp(-k * (t - t0))))


def conservative(t, rate=0.18):
    return np.minimum(1.0, rate * t)


def late_start(t, delay=2.0, rate=0.18):
    return np.where(t < delay, 0.0, np.minimum(1.0, rate * (t - delay)))


def linear_years(n_years):
    """A linear schedule completing in exactly n_years."""
    return lambda t: np.minimum(1.0, t / float(n_years))


SCENARIOS = {
    "Aggressive": aggressive,
    "Conservative": conservative,
    "Late Start": late_start,
}
