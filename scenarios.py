"""
scenarios.py

Deployment coverage functions L(t), implementing the three migration
strategies described in the paper (Sec. scenarios, Eqs. 16-18 in the
original draft numbering). L(t) is bounded to [0, 1] since it represents
the fraction of migrated systems.

All functions accept a NumPy array of years `t` and return an array of
the same shape, so they vectorize cleanly across the (N_iterations,
T_years) simulation grid.
"""

import numpy as np


def aggressive_logistic(t: np.ndarray, L_max: float, k: float, t0: float) -> np.ndarray:
    """
    Implements Eq. scenarios-aggressive:
        L(t) = min(1.0, L_max / (1 + exp(-k*(t - t0))))

    Front-loaded adoption curve: slow start, rapid mid-period ramp,
    saturating near L_max.
    """
    logistic = L_max / (1.0 + np.exp(-k * (t - t0)))
    return np.minimum(1.0, logistic)


def conservative_linear(t: np.ndarray, rate: float) -> np.ndarray:
    """
    Implements Eq. scenarios-conservative:
        L(t) = min(1.0, rate * t)

    Steady incremental migration at a constant annual rate.
    """
    return np.minimum(1.0, rate * t)


def late_start(t: np.ndarray, t_delay: float, rate: float) -> np.ndarray:
    """
    Implements Eq. scenarios-late-start:
        L(t) = 0                          for t < t_delay
        L(t) = min(1.0, rate*(t - t_delay)) for t >= t_delay

    Same linear rate as the conservative scenario, but migration does
    not begin until t_delay years have passed.
    """
    coverage = np.where(
        t < t_delay,
        0.0,
        np.minimum(1.0, rate * (t - t_delay)),
    )
    return coverage


def get_all_scenarios(t: np.ndarray, cfg) -> dict:
    """
    Convenience wrapper returning all three scenarios evaluated at the
    given years, using parameters from a SimulationConfig instance.
    """
    return {
        "Aggressive": aggressive_logistic(
            t, cfg.AGGRESSIVE_LMAX, cfg.AGGRESSIVE_K, cfg.AGGRESSIVE_T0
        ),
        "Conservative": conservative_linear(t, cfg.CONSERVATIVE_RATE),
        "Late Start": late_start(t, cfg.LATE_START_DELAY, cfg.LATE_START_RATE),
    }
