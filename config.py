"""
config.py

Central configuration for the PQC migration risk Monte Carlo simulation.

------------------------------------------------------------------------
KNOWN OPEN ISSUE -- READ BEFORE EDITING C_ALG BELOW
------------------------------------------------------------------------
The adversarial capability variable A(t) in the model (see simulation.py,
`log_capability`) represents QUANTUM adversarial capability, measured in
quantum logical operations / gates. For the capability gap
delta(t) = log10(A(t)) - C_alg to mean anything, C_alg MUST be measured
in the same computational model as A(t) -- i.e., you should generally be
using the "quantum" cost, not the "classical" cost, for a given
algorithm, unless you are deliberately running a separate classical-
threat-only scenario.

Mixing them (e.g. comparing a quantum A(t) against a classical GNFS-
derived C_alg) is a unit-consistency error, not a modeling choice. This
was the central bug in the earlier draft of this work. The simulation
code will not silently guess for you -- see `get_cost()` below, which
requires you to explicitly say which cost type you want and will raise
if that value hasn't been filled in yet.
------------------------------------------------------------------------
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SimulationConfig:
    # --- Monte Carlo controls ---
    # N=5000 because the convergence sweep (convergence.py, paper Sec.
    # Convergence) shows N=1000 sits ~0.033 outside the paper's stated
    # 0.01 tolerance on mean RQR at Year 15; N=5000 is the smallest N
    # within tolerance and is what every headline number in the paper
    # (main.py, run_sensitivity.py) is actually generated at.
    N_ITERATIONS: int = 5000
    T_YEARS: int = 15
    RANDOM_SEED: int = 42

    # --- Adversarial capability growth model (Eq. capability-growth) ---
    # log10(A(t)) = log10(A0) + g*t + eps_t
    A0: float = 1e5          # initial adversarial capability, operations

    # Growth-rate distribution for g. "gamma" is the paper's calibrated
    # default (Sec. growth); "gaussian" is retained ONLY for direct
    # comparison against the earlier (rejected) Gaussian fit -- it is
    # not an alternative default, per paper Sec. growth.
    GROWTH_MODEL: str = "gamma"

    # Gamma(k, theta) calibrated via two-target root-finding against
    # GRI/evolutionQ 2024 survey midpoints (0.095 @ 5yr, 0.265 @ 10yr).
    # E[g] = k*theta ~= 0.370, sd(g) ~= 0.451, Pr(g<0)=0 by construction.
    GAMMA_K: float = 0.6749
    GAMMA_THETA: float = 0.5485

    # Gaussian g ~ N(MU_G, SIGMA_G^2), calibrated to the SAME two survey
    # targets. Kept only to reproduce the paper's "discovered problem
    # with the Gaussian assumption" comparison (Pr(g<0) ~= 0.478 under
    # this fit) -- do not use as the simulation default.
    MU_G: float = 0.0385
    SIGMA_G: float = 0.7047

    SIGMA_EPS: float = 0.05  # std dev of annual noise term (still ILLUSTRATIVE -- paper Sec. growth "What Remains Illustrative")

    # --- Residual Quantum Risk logistic sensitivity (Eq. rqr) ---
    ALPHA: float = 2.0       # calibrated from an assumed anchor p(delta=1)=0.88
                              # (ILLUSTRATIVE anchor, not an elicited empirical value)

    # --- CAS weights (Eq. cas) -- must sum to 1.0, validated at load time ---
    W_AS: float = 0.40
    W_KM: float = 0.25
    W_DC: float = 0.20
    W_CAI: float = 0.15

    # --- Placeholder organizational metrics (see paper Sec. cas, M_KM / M_CAI) ---
    # These are NOT derived from any organization's real telemetry yet.
    # Replace with measured values (HSM coverage %, key rotation vs.
    # policy, CBOM completeness %, etc.) before treating results as
    # anything more than illustrative.
    M_KM_CONSTANT: float = 0.85   # placeholder, flagged in paper as needing derivation
    M_CAI_CONSTANT: float = 0.75  # placeholder

    # --- CAS threshold used for scenario labeling / figure reference line ---
    CAS_MIN: float = 0.70

    # --- Deployment scenario parameters (Eq. scenarios) ---
    AGGRESSIVE_LMAX: float = 0.98
    AGGRESSIVE_K: float = 1.2
    AGGRESSIVE_T0: float = 3.0
    CONSERVATIVE_RATE: float = 0.18
    LATE_START_DELAY: float = 2.0
    LATE_START_RATE: float = 0.18


# ------------------------------------------------------------------------
# Attack cost table: log10(operations) or log10(gates), by algorithm and
# cost type. Values are as cited in the paper draft (Sec. costs / Table I)
# where available. `None` means "not yet sourced -- do not simulate this
# combination until filled in."
# ------------------------------------------------------------------------
ATTACK_COSTS = {
    "RSA-2048": {
        "classical": 33.7,   # 2^112, NIST SP 800-57 GNFS-based security strength [nist80057]
        "quantum": 9.81,     # ~6.5e9 Toffoli gates, Gidney 2025 [gidney2025];
                              # original estimate ~2.7e9 gates / 20M qubits / 8h in
                              # Gidney & Ekera 2021 [gidney2021]. The 2021->2025 revision
                              # is itself evidence that C_alg is NOT static -- see
                              # Limitations, "Attack Cost Stationarity".
    },
    "ECC-P256": {
        "classical": 38.5,   # ~2^128, NIST SP 800-57 [nist80057]
        "quantum": 7.9,      # ~8e7 Toffoli gates (midpoint of 70-90M), Google Quantum AI
                              # 2026 whitepaper [googleqai2026], for secp256k1 (same 256-bit
                              # curve order as P-256, but a DIFFERENT curve -- treated here as
                              # a proxy, not an exact P-256-specific estimate).
                              # CONTRAST: Roetteler et al. 2017 [roetteler2017] estimated
                              # ~1.3e11 Toffoli gates for P-256 directly -- a ~1500x reduction
                              # over 9 years. Cite both; the gap IS the finding.
    },
    "Kyber-512": {
        "classical": 35.5,   # 118-bit core-SVP classical, Kyber round-3 spec Table 4 [kyberspec]
        "quantum": 32.2,     # 107-bit core-SVP quantum, Kyber round-3 spec Table 4 [kyberspec].
                              # DISPUTED: NIST PQC forum official comments [kyberofficialcomment]
                              # argue the true classical hardness may be substantially higher
                              # (~2^143 "gates") once communication/memory costs are counted --
                              # meaning Kyber-512 may be SAFER than this number implies. Report
                              # the spec's own number, but state the dispute in the paper text,
                              # do not present this as an uncontested figure.
    },
    "Kyber-768": {
        "classical": 54.8,   # 182-bit core-SVP classical, Kyber round-3 spec Table 4 [kyberspec]
        "quantum": 49.7,     # 165-bit core-SVP quantum, Kyber round-3 spec Table 4 [kyberspec]
    },
    "Kyber-1024": {
        "classical": 77.1,   # 256-bit core-SVP classical, Kyber round-3 spec Table 4 [kyberspec]
        "quantum": 69.8,     # 232-bit core-SVP quantum, Kyber round-3 spec Table 4 [kyberspec]
    },
}


class MissingCostError(ValueError):
    """Raised when a requested attack-cost value has not been sourced yet."""
    pass


def get_cost(algorithm: str, cost_type: str) -> float:
    """
    Look up C_alg for a given algorithm and cost type ("classical" or
    "quantum"). Raises MissingCostError if the value is None (not yet
    sourced) rather than silently defaulting to 0 or skipping it --
    a None value here means "the underlying research isn't finished",
    and treating it as zero would produce a meaningless but
    plausible-looking result.
    """
    if algorithm not in ATTACK_COSTS:
        raise KeyError(
            f"Unknown algorithm '{algorithm}'. "
            f"Available: {list(ATTACK_COSTS.keys())}"
        )
    if cost_type not in ("classical", "quantum"):
        raise ValueError("cost_type must be 'classical' or 'quantum'")

    value = ATTACK_COSTS[algorithm][cost_type]
    if value is None:
        raise MissingCostError(
            f"No '{cost_type}' attack cost has been sourced yet for "
            f"'{algorithm}'. Fill in config.ATTACK_COSTS with a cited "
            f"value before simulating this combination -- do not guess."
        )
    return value


def validate_config(cfg: SimulationConfig) -> None:
    """Sanity-check config values before running anything."""
    weight_sum = cfg.W_AS + cfg.W_KM + cfg.W_DC + cfg.W_CAI
    if abs(weight_sum - 1.0) > 1e-9:
        raise ValueError(
            f"CAS weights must sum to 1.0, got {weight_sum:.6f} "
            f"(W_AS={cfg.W_AS}, W_KM={cfg.W_KM}, W_DC={cfg.W_DC}, W_CAI={cfg.W_CAI})"
        )
    if cfg.N_ITERATIONS <= 0 or cfg.T_YEARS <= 0:
        raise ValueError("N_ITERATIONS and T_YEARS must be positive integers.")
    if cfg.SIGMA_G < 0 or cfg.SIGMA_EPS < 0:
        raise ValueError("Standard deviations cannot be negative.")
    if cfg.GROWTH_MODEL not in ("gamma", "gaussian"):
        raise ValueError(
            f"GROWTH_MODEL must be 'gamma' or 'gaussian', got '{cfg.GROWTH_MODEL}'. "
            f"'gamma' is the paper's calibrated default; 'gaussian' exists only "
            f"for the Sec. growth comparison."
        )
    if cfg.GROWTH_MODEL == "gamma" and (cfg.GAMMA_K <= 0 or cfg.GAMMA_THETA <= 0):
        raise ValueError("GAMMA_K and GAMMA_THETA must be positive.")


# Algorithms to actually run through the simulation, and which cost type
# to use for each. This list is separate from ATTACK_COSTS so you can
# add algorithms to the cost table (for reference / future work) without
# forcing them into the simulation before they're ready.
#
# IMPORTANT: cost_type is "quantum" by default, per the unit-consistency
# note at the top of this file. Only RSA-2048 currently has a sourced
# quantum value; Kyber-512's quantum value is still None (disputed /
# unresolved), so it will raise MissingCostError until filled in -- this
# is intentional, not a bug.
SIMULATION_TARGETS = [
    {"algorithm": "RSA-2048", "cost_type": "quantum"},
    {"algorithm": "ECC-P256", "cost_type": "quantum"},
    {"algorithm": "Kyber-512", "cost_type": "quantum"},
    {"algorithm": "Kyber-768", "cost_type": "quantum"},
    {"algorithm": "Kyber-1024", "cost_type": "quantum"},
]
