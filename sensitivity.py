"""
sensitivity.py

Sensitivity analysis for the PQC migration risk model (paper Section
XIII). This module sweeps individual parameters and reports the
resulting change in TCI, and produces a 2D heatmap over two parameters
jointly, matching the structure of the original paper's Fig. 3.

------------------------------------------------------------------------
SCOPE NOTE -- read before extending this file
------------------------------------------------------------------------
This module covers sensitivity to mu_g, alpha, and w_AS, because these
are the parameters actually wired into the current model (see
simulation.py). The original paper draft also claimed a "Sensitivity
to Hybrid Overhead" analysis; that axis is NOT implemented here,
because the current CAS model has no overhead parameter to sweep --
M_CAI is a constant (see config.py, M_CAI_CONSTANT). Implementing that
sensitivity axis honestly requires first extending composite_assurance
_score() (or a new metric function) so that M_CAI is actually a
function of hybrid handshake overhead, with a cited relationship
(e.g., from Sikeridis et al. 2020 / Paquin et al. 2020's measured
20-80% latency increase). Until that modeling work is done, running a
sensitivity sweep on "overhead" would be sweeping a parameter that
does nothing -- which is worse than not reporting it at all. See
`hybrid_overhead_not_implemented()` below, which raises with this
explanation rather than silently returning a fake result.
------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd

import simulation
import scenarios


def scan_ecc_cost_model(cfg, L_t, seed) -> pd.DataFrame:
    """
    Discrete (not swept) sensitivity check: ECC-P256's quantum cost
    under the min-gate secp256k1 proxy (this paper's default,
    C_alg=7.90) versus the min-qubit P-256-specific 2026 estimate
    (C_alg=38.10*log10(2)=~11.47, Chevignard et al. EUROCRYPT 2026).
    This is the single sensitivity the paper's own prose already
    identifies as capable of reversing the ECC-vs-RSA headline finding,
    so it belongs in the same table as the continuous parameter
    sweeps, not left as a prose-only caveat.
    """
    rows = []
    for label, C_alg in [("min-gate (paper default)", 7.90),
                          ("min-qubit (Chevignard 2026)", 38.10 * np.log10(2))]:
        rng = np.random.default_rng(seed)
        result = simulation.run_cas_simulation(cfg, C_alg, L_t, rng)
        rows.append({"cost_model": label, "C_alg": C_alg,
                      "tci_mean": result["tci"].mean(),
                      "tci_std": result["tci"].std()})
    return pd.DataFrame(rows)


def _run_tci_for_params(cfg, C_alg, L_t, seed, mu_g=None, sigma_g=None,
                         alpha=None, w_as=None, w_km=None, w_dc=None, w_cai=None,
                         growth_model=None, gamma_k=None, gamma_theta=None):
    """
    Run one Monte Carlo TCI simulation with a subset of parameters
    overridden from their defaults in `cfg`. Returns (tci_mean, tci_std).

    This does not mutate `cfg` -- a fresh SimulationConfig-like object
    is built for each call so sweeps don't leak state between runs.

    Growth-rate sensitivity (paper Sec. sensitivity) sweeps under the
    calibrated Gamma model by varying gamma_theta with gamma_k held
    fixed, NOT by overriding a Gaussian mu_g -- see scan_mu_g below,
    which converts a target E[g] into the corresponding theta.
    """
    import config as config_module

    overrides = dict(
        N_ITERATIONS=cfg.N_ITERATIONS,
        T_YEARS=cfg.T_YEARS,
        RANDOM_SEED=seed,
        A0=cfg.A0,
        GROWTH_MODEL=growth_model if growth_model is not None else cfg.GROWTH_MODEL,
        GAMMA_K=gamma_k if gamma_k is not None else cfg.GAMMA_K,
        GAMMA_THETA=gamma_theta if gamma_theta is not None else cfg.GAMMA_THETA,
        MU_G=mu_g if mu_g is not None else cfg.MU_G,
        SIGMA_G=sigma_g if sigma_g is not None else cfg.SIGMA_G,
        SIGMA_EPS=cfg.SIGMA_EPS,
        ALPHA=alpha if alpha is not None else cfg.ALPHA,
        W_AS=w_as if w_as is not None else cfg.W_AS,
        W_KM=w_km if w_km is not None else cfg.W_KM,
        W_DC=w_dc if w_dc is not None else cfg.W_DC,
        W_CAI=w_cai if w_cai is not None else cfg.W_CAI,
        M_KM_CONSTANT=cfg.M_KM_CONSTANT,
        M_CAI_CONSTANT=cfg.M_CAI_CONSTANT,
        CAS_MIN=cfg.CAS_MIN,
    )
    swept_cfg = config_module.SimulationConfig(**overrides)
    config_module.validate_config(swept_cfg)

    rng = np.random.default_rng(seed)
    result = simulation.run_cas_simulation(swept_cfg, C_alg, L_t, rng)
    return result["tci"].mean(), result["tci"].std()


def scan_mu_g(cfg, C_alg, L_t, mu_g_range, seed) -> pd.DataFrame:
    """
    Sweep the growth-rate mean E[g] across `mu_g_range`, holding all
    other parameters at their cfg defaults.

    Under the paper's calibrated Gamma default (Sec. sensitivity,
    "growth-rate sensitivity is now expressed as a sweep over E[g] ...
    by holding shape k fixed and adjusting scale theta"), each target
    E[g] is converted to theta = E[g] / GAMMA_K before running.
    Under the legacy Gaussian comparison model, mu_g_range is used
    directly as the Gaussian mean.

    Returns a DataFrame with columns ['mu_g', 'tci_mean', 'tci_std'],
    where 'mu_g' holds the target E[g] regardless of which underlying
    distribution produced it.
    """
    rows = []
    for target_mean in mu_g_range:
        if cfg.GROWTH_MODEL == "gamma":
            theta = target_mean / cfg.GAMMA_K
            tci_mean, tci_std = _run_tci_for_params(
                cfg, C_alg, L_t, seed, gamma_theta=theta
            )
        else:
            tci_mean, tci_std = _run_tci_for_params(cfg, C_alg, L_t, seed, mu_g=target_mean)
        rows.append({"mu_g": target_mean, "tci_mean": tci_mean, "tci_std": tci_std})
    return pd.DataFrame(rows)


def scan_alpha(cfg, C_alg, L_t, alpha_range, seed) -> pd.DataFrame:
    """
    Sweep alpha (logistic sensitivity parameter), holding all other
    parameters at their cfg defaults. Returns a DataFrame with columns
    ['alpha', 'tci_mean', 'tci_std'].
    """
    rows = []
    for alpha in alpha_range:
        tci_mean, tci_std = _run_tci_for_params(cfg, C_alg, L_t, seed, alpha=alpha)
        rows.append({"alpha": alpha, "tci_mean": tci_mean, "tci_std": tci_std})
    return pd.DataFrame(rows)


def scan_w_as(cfg, C_alg, L_t, w_as_range, seed) -> pd.DataFrame:
    """
    Sweep w_AS (Algorithm Strength weight in CAS), redistributing the
    remainder proportionally across w_KM, w_DC, w_CAI so weights
    continue to sum to 1.0. This matches how a real policy change
    would work -- increasing emphasis on algorithm strength has to
    come from somewhere else in the weight budget, not from thin air.

    Returns a DataFrame with columns ['w_as', 'tci_mean', 'tci_std'].
    """
    rows = []
    base_remainder = cfg.W_KM + cfg.W_DC + cfg.W_CAI
    for w_as in w_as_range:
        remaining_budget = 1.0 - w_as
        scale = remaining_budget / base_remainder if base_remainder > 0 else 0
        w_km = cfg.W_KM * scale
        w_dc = cfg.W_DC * scale
        w_cai = cfg.W_CAI * scale
        tci_mean, tci_std = _run_tci_for_params(
            cfg, C_alg, L_t, seed, w_as=w_as, w_km=w_km, w_dc=w_dc, w_cai=w_cai
        )
        rows.append({"w_as": w_as, "tci_mean": tci_mean, "tci_std": tci_std})
    return pd.DataFrame(rows)


def joint_heatmap(cfg, C_alg, L_t, mu_g_range, w_as_range, seed) -> np.ndarray:
    """
    Compute a 2D grid of TCI mean values over (mu_g, w_AS), matching
    the structure of the original paper's Fig. 3 heatmap. Rows index
    w_as_range, columns index mu_g_range (grid[i, j] = TCI at
    w_as_range[i], mu_g_range[j]).

    NOTE: this is O(len(mu_g_range) * len(w_as_range)) full Monte Carlo
    runs. With N=1000 and a 10x10 grid this is 100 simulation runs --
    still well under a second total on typical hardware, but be aware
    the cost scales with grid resolution if you increase it.
    """
    base_remainder = cfg.W_KM + cfg.W_DC + cfg.W_CAI
    grid = np.zeros((len(w_as_range), len(mu_g_range)))

    for i, w_as in enumerate(w_as_range):
        remaining_budget = 1.0 - w_as
        scale = remaining_budget / base_remainder if base_remainder > 0 else 0
        w_km = cfg.W_KM * scale
        w_dc = cfg.W_DC * scale
        w_cai = cfg.W_CAI * scale
        for j, target_mean in enumerate(mu_g_range):
            if cfg.GROWTH_MODEL == "gamma":
                theta = target_mean / cfg.GAMMA_K
                tci_mean, _ = _run_tci_for_params(
                    cfg, C_alg, L_t, seed, gamma_theta=theta,
                    w_as=w_as, w_km=w_km, w_dc=w_dc, w_cai=w_cai
                )
            else:
                tci_mean, _ = _run_tci_for_params(
                    cfg, C_alg, L_t, seed, mu_g=target_mean,
                    w_as=w_as, w_km=w_km, w_dc=w_dc, w_cai=w_cai
                )
            grid[i, j] = tci_mean

    return grid


def hybrid_overhead_not_implemented():
    """
    Intentionally raises. See the SCOPE NOTE at the top of this file:
    sensitivity to hybrid deployment overhead cannot be honestly
    computed until M_CAI is a function of overhead rather than a
    constant. Do not fill this in with a fabricated relationship just
    to complete the sensitivity table -- extend
    simulation.composite_assurance_score() properly first, citing
    Sikeridis et al. 2020 / Paquin et al. 2020 for the overhead-latency
    relationship, then implement the sweep here.
    """
    raise NotImplementedError(
        "Hybrid overhead sensitivity requires M_CAI to be modeled as a "
        "function of overhead, not a constant. Not implemented -- see "
        "module docstring."
    )


def summarize_sensitivity(mu_g_df: pd.DataFrame, alpha_df: pd.DataFrame,
                           w_as_df: pd.DataFrame, ecc_cost_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Build a tornado-style summary table: for each swept parameter,
    report the range tested and the resulting delta in TCI mean
    (max - min across the sweep). This is the actual number that
    belongs in the paper's Section XIII prose (e.g. "For mu_g in
    [0.3, 0.7]: delta_TCI = ...") -- computed from real sweep output,
    not asserted.

    ecc_cost_df, if provided, adds a fourth, discrete (not swept)
    entry: the ECC-P256 min-gate-vs-min-qubit cost-model choice
    (see scan_ecc_cost_model). It is reported the same way as the
    continuous sweeps -- range/min/max/delta -- even though "range"
    here means two discrete options rather than an interval.
    """
    rows = []
    for name, df, param_col in [
        ("mu_g", mu_g_df, "mu_g"),
        ("alpha", alpha_df, "alpha"),
        ("w_AS", w_as_df, "w_as"),
    ]:
        delta = df["tci_mean"].max() - df["tci_mean"].min()
        rows.append({
            "parameter": name,
            "range_min": df[param_col].min(),
            "range_max": df[param_col].max(),
            "tci_min": df["tci_mean"].min(),
            "tci_max": df["tci_mean"].max(),
            "delta_tci": delta,
        })
    if ecc_cost_df is not None:
        delta = ecc_cost_df["tci_mean"].max() - ecc_cost_df["tci_mean"].min()
        rows.append({
            "parameter": "ECC_cost_model",
            "range_min": ecc_cost_df["C_alg"].min(),
            "range_max": ecc_cost_df["C_alg"].max(),
            "tci_min": ecc_cost_df["tci_mean"].min(),
            "tci_max": ecc_cost_df["tci_mean"].max(),
            "delta_tci": delta,
        })
    return pd.DataFrame(rows)
