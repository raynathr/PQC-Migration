"""
baselines.py

Quantitative comparison against simpler alternatives (Reviewer 2, major
concern 5). The question is not whether the stochastic framework is more
elaborate -- it obviously is -- but whether it produces decision
information the simpler methods do not.

Five methods are run on identical inputs:

  B1  Mosca X + Y > Z, deterministic. X = median CRQC arrival year under
      the calibrated growth model; Y = migration duration; Z = data
      shelf life. Verdict: act now / do not act now.
  B2  Qualitative five-level readiness label (CSA-style) mapped from
      migration coverage at the assessment year.
  B3  Deterministic RQR: growth rate fixed at E[ghat], no uncertainty
      anywhere, exposure coupling retained.
  B4  CAS without uncertainty: B3's point RQR fed through the full CAS
      aggregation.
  B5  The full stochastic model.

Each method is scored on what it can and cannot discriminate:
per-asset-class prioritisation, scenario ranking, and whether it
expresses any uncertainty at all.
"""

import numpy as np


def mosca_verdict(x_years, y_years, z_years):
    """X + Y > Z -> the organization is already late."""
    return (x_years + y_years) > z_years


def median_crqc_year(khat, thetahat, d_alg, t_max=60):
    """
    Median arrival year for capability sufficient to break an algorithm
    with relative deficit d_alg: the t at which Pr(ghat*t >= d_alg) = 0.5,
    i.e. t = d_alg / median(ghat).
    """
    from scipy.stats import gamma as gd
    med_g = gd.ppf(0.5, a=khat, scale=thetahat)
    if med_g <= 0:
        return np.inf
    return min(d_alg / med_g, t_max)


def readiness_level(coverage):
    """CSA-style five-level maturity label from coverage."""
    edges = [0.05, 0.25, 0.60, 0.90]
    labels = ["Initial", "Developing", "Defined", "Managed", "Optimized"]
    return labels[int(np.searchsorted(edges, coverage))]


def discrimination(values, tol=1e-9):
    """
    How many distinct decisions a method yields over a set of inputs.
    A method that returns the same verdict for every asset class cannot
    be used to order them, whatever its other merits.
    """
    vals = [round(float(v), 6) if isinstance(v, (int, float, np.floating)) else v
            for v in values]
    return len(set(vals))
