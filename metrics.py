"""
metrics.py

Operational definitions for the two CAS components that the reviewers
identified as illustrative constants: key-management posture (M_KM) and
the crypto-agility index (M_CAI).

Each is now a weighted aggregate of NAMED, MEASURABLE indicators, every
one of which an organization can read off its own telemetry. Each
indicator carries:

  * a baseline value at t = 0,
  * an attainable ceiling,
  * a migration-coupling coefficient kappa in [0, 1] saying how much of
    the gap to the ceiling is closed by migration progress L(t)
    (kappa = 0 -> the indicator is unaffected by migrating; kappa = 1 ->
    it reaches its ceiling exactly when migration completes),
  * a provenance tag: "survey" (read off a cited public survey),
    "derived" (computed from a cited survey figure), or "elicited-open"
    (still an assumption awaiting the expert-elicitation exercise in
    Appendix E, and swept in the sensitivity analysis).

Two things follow that the constant-valued version could not deliver.
First, M_KM and M_CAI become time-varying and migration-coupled, so
they respond to migration strategy instead of shifting every scenario
by the same additive amount. Second, the hybrid-deployment-overhead
sensitivity that the previous version reported as "not yet testable"
becomes testable, because indicator CA5 is now an explicit function of
measured hybrid handshake overhead.
"""

from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass(frozen=True)
class Indicator:
    key: str
    label: str
    weight: float
    baseline: float
    ceiling: float
    kappa: float
    provenance: str
    source: str


# ---------------------------------------------------------------------
# M_KM -- key-management posture
# ---------------------------------------------------------------------
M_KM_INDICATORS_TYPICAL = [
    Indicator("KM1", "HSM-backed key coverage", 0.25, 0.42, 0.95, 0.45,
              "elicited-open", "assumed; bracketed 0.30-0.55 in sweep"),
    Indicator("KM2", "Key-rotation compliance vs. policy", 0.20, 0.50, 0.95, 0.35,
              "elicited-open", "assumed; bracketed 0.35-0.65 in sweep"),
    Indicator("KM3", "Key-age compliance (no key past max age)", 0.15, 0.55, 0.95, 0.30,
              "elicited-open", "assumed; bracketed 0.40-0.70 in sweep"),
    Indicator("KM4", "Absence of weak/legacy keys", 0.15, 0.70, 1.00, 0.60,
              "elicited-open", "assumed; bracketed 0.55-0.85 in sweep"),
    Indicator("KM5", "Key and certificate inventory completeness", 0.25, 0.28, 0.98, 0.70,
              "survey", "sectigo2025: 28% report a complete certificate inventory"),
]

# ---------------------------------------------------------------------
# M_CAI -- crypto-agility index
# ---------------------------------------------------------------------
M_CAI_INDICATORS_TYPICAL = [
    Indicator("CA1", "CBOM completeness", 0.25, 0.44, 0.98, 0.65,
              "survey", "digicert2026: 44% have completed a cryptographic inventory"),
    Indicator("CA2", "Algorithm replacement capability", 0.20, 0.45, 0.95, 0.75,
              "derived", "digicert2026: 45% have a documented transition plan (proxy)"),
    Indicator("CA3", "Cryptographic dependency discovery coverage", 0.15, 0.40, 0.95, 0.60,
              "elicited-open", "assumed; bracketed 0.25-0.55 in sweep"),
    Indicator("CA4", "Patch/update latency score", 0.10, 0.55, 0.90, 0.20,
              "elicited-open", "assumed; bracketed 0.40-0.70 in sweep"),
    Indicator("CA5", "Hybrid-PQC deployment capability", 0.20, 0.20, 0.95, 0.90,
              "derived", "overhead-discounted; sikeridis2020, paquin2020"),
    Indicator("CA6", "Certificate replacement time score", 0.10, 0.45, 0.95, 0.50,
              "elicited-open", "assumed; bracketed 0.30-0.60 in sweep"),
]

# The "mature / target-state" profile: the posture the original draft's
# constants M_KM = 0.85 and M_CAI = 0.75 were standing in for. Retained so
# the revision can report both profiles rather than silently swapping one
# assumption for another.
def _mature(inds, lift=0.42):
    return [Indicator(i.key, i.label, i.weight,
                       min(i.ceiling, i.baseline + lift * (i.ceiling - i.baseline)),
                       i.ceiling, i.kappa, i.provenance, i.source + " [mature profile]")
            for i in inds]


M_KM_INDICATORS_MATURE = _mature(M_KM_INDICATORS_TYPICAL, lift=0.78)
M_CAI_INDICATORS_MATURE = _mature(M_CAI_INDICATORS_TYPICAL, lift=0.62)

# Measured hybrid TLS 1.3 handshake latency overhead, as a fraction.
# Cited range is 20-80% (sikeridis2020, paquin2020); midpoint default.
HYBRID_OVERHEAD_RANGE = (0.20, 0.80)
HYBRID_OVERHEAD_DEFAULT = 0.50
# Tolerance coefficient: how much of CA5's attainable value a unit of
# handshake overhead removes. Elicitation-open; swept.
OVERHEAD_TOLERANCE = 0.25


def indicator_trajectory(ind: Indicator, L_t: np.ndarray) -> np.ndarray:
    """x_i(t) = baseline + kappa_i * (ceiling - baseline) * L(t)."""
    return ind.baseline + ind.kappa * (ind.ceiling - ind.baseline) * L_t


def aggregate(indicators, L_t: np.ndarray, hybrid_overhead: float = None,
              overhead_tolerance: float = OVERHEAD_TOLERANCE) -> np.ndarray:
    """
    Weighted aggregate of an indicator set over a coverage trajectory.
    Weights are renormalized defensively so a partial indicator set still
    produces a score on [0, 1].
    """
    total_w = sum(i.weight for i in indicators)
    out = np.zeros_like(np.asarray(L_t, dtype=float))
    for ind in indicators:
        x = indicator_trajectory(ind, L_t)
        if ind.key == "CA5" and hybrid_overhead is not None:
            # Hybrid capability is discounted by measured handshake overhead:
            # a deployment that is technically possible but 80% slower is
            # not equally deployable. This is the dependency whose absence
            # previously made the overhead sensitivity untestable.
            x = x * (1.0 - overhead_tolerance * hybrid_overhead)
        out = out + (ind.weight / total_w) * x
    return out


def blend(typical, mature, s: float):
    """
    Interpolate indicator baselines between the industry-typical (s=0) and
    mature (s=1) profiles. Used for the posture axis of the 2D sensitivity
    surface, where a two-valued profile flag would give a two-row heatmap.
    Only the baseline moves; ceilings and coupling coefficients are
    properties of the indicator, not of the organization.
    """
    out = []
    for a, b in zip(typical, mature):
        out.append(Indicator(a.key, a.label, a.weight,
                             a.baseline + s * (b.baseline - a.baseline),
                             a.ceiling, a.kappa, a.provenance, a.source))
    return out


def _km_set(profile):
    if isinstance(profile, (int, float)):
        return blend(M_KM_INDICATORS_TYPICAL, M_KM_INDICATORS_MATURE, float(profile))
    return M_KM_INDICATORS_TYPICAL if profile == "typical" else M_KM_INDICATORS_MATURE


def _cai_set(profile):
    if isinstance(profile, (int, float)):
        return blend(M_CAI_INDICATORS_TYPICAL, M_CAI_INDICATORS_MATURE, float(profile))
    return M_CAI_INDICATORS_TYPICAL if profile == "typical" else M_CAI_INDICATORS_MATURE


def m_km(L_t, profile="typical"):
    return aggregate(_km_set(profile), L_t)


def m_cai(L_t, profile="typical", hybrid_overhead=HYBRID_OVERHEAD_DEFAULT,
          overhead_tolerance=OVERHEAD_TOLERANCE):
    return aggregate(_cai_set(profile), L_t, hybrid_overhead=hybrid_overhead,
                      overhead_tolerance=overhead_tolerance)


def indicator_table(profile="typical"):
    inds = (M_KM_INDICATORS_TYPICAL + M_CAI_INDICATORS_TYPICAL if profile == "typical"
            else M_KM_INDICATORS_MATURE + M_CAI_INDICATORS_MATURE)
    return [{"key": i.key, "label": i.label, "w": i.weight,
             "baseline": round(i.baseline, 3), "ceiling": i.ceiling,
             "kappa": i.kappa, "provenance": i.provenance, "source": i.source}
            for i in inds]


if __name__ == "__main__":
    L0 = np.array([0.0]); L1 = np.array([1.0])
    for prof in ("typical", "mature"):
        print(f"{prof:8s}  M_KM(L=0)={m_km(L0, prof)[0]:.3f} -> M_KM(L=1)={m_km(L1, prof)[0]:.3f}"
              f"   M_CAI(L=0)={m_cai(L0, prof)[0]:.3f} -> M_CAI(L=1)={m_cai(L1, prof)[0]:.3f}")
