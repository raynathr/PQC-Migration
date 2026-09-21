"""
costs.py

Probabilistic quantum attack cost C_alg (paper Sec. "Attack Cost as a
Random Variable").

Motivation (Editor comment 6, Reviewer 1 comment 4): the manuscript's
own sensitivity analysis showed that the ECC-vs-RSA ordering reverses
depending on whether a minimum-gate or a minimum-qubit resource
estimate is used, with dTCI = 0.075 -- the second-largest modeled
sensitivity. A quantity that can flip the paper's headline finding
should not be a deterministic scalar.

We therefore represent C_alg as a discrete random variable supported on
the PUBLISHED resource estimates for that algorithm, with weights
reflecting how directly each estimate applies to the primitive and
cost convention in question. Nothing here is invented: every support
point is a cited number. The weights are the elicitable part, and are
swept in the sensitivity analysis.

Each estimate additionally carries its position on the two axes that
the resource-estimation literature actually trades off -- logical
qubits and Toffoli-gate count -- so the same table supports the
multidimensional reading requested by the Editor without forcing it
into the scalar pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

LOG10_2 = np.log10(2.0)


@dataclass(frozen=True)
class ResourceEstimate:
    label: str
    log10_gates: float          # log10 Toffoli / T-gate count
    log10_qubits: Optional[float]  # log10 logical qubits, where published
    convention: str             # "min-gate" | "min-qubit" | "core-SVP"
    citation: str
    weight: float               # prior mass, see module docstring
    exact_primitive: bool       # True if computed for THIS primitive, not a proxy


# --- RSA-2048 -----------------------------------------------------------
# Gidney & Ekera 2021: ~2.7e9 Toffoli, 20M physical qubits.
# Gidney 2025:         ~6.5e9 Toffoli, <1M physical qubits.
# Both are min-gate-family surface-code designs; the 2025 revision is the
# more recent and is weighted accordingly.
RSA2048 = [
    # Logical-qubit counts are left None for RSA: both sources report
    # PHYSICAL qubit budgets (20M in 2021, <1M in 2025), which are not
    # comparable with the logical-qubit figures published for the ECC
    # circuits. We do not convert between the two.
    ResourceEstimate("Gidney 2025", np.log10(6.5e9), None,
                      "min-gate", "gidney2025", 0.60, True),
    ResourceEstimate("Gidney & Ekera 2021", np.log10(2.7e9), None,
                      "min-gate", "gidney2021", 0.40, True),
]

# --- ECC P-256 ----------------------------------------------------------
# The three published points disagree by ~3.6 orders of magnitude and sit
# at opposite ends of the qubit/gate trade-off. This spread IS the finding.
ECCP256 = [
    ResourceEstimate("Google QAI 2026 (secp256k1 proxy)", np.log10(8.0e7), None,
                      "min-gate", "googleqai2026", 0.35, False),
    ResourceEstimate("Roetteler et al. 2017 (P-256)", np.log10(1.3e11), np.log10(2330.0),
                      "min-gate", "roetteler2017", 0.30, True),
    ResourceEstimate("Chevignard et al. 2026 (P-256)", 38.10 * LOG10_2, np.log10(1098.0),
                      "min-qubit", "chevignard2026", 0.35, True),
]

# --- Kyber / ML-KEM -----------------------------------------------------
# Spec core-SVP quantum figures, plus the NIST PQC-forum position that the
# true cost is substantially higher once memory/communication is counted.
# The dispute is one-sided (it argues Kyber is SAFER), so the second
# support point sits above the spec value, never below it.
KYBER512 = [
    ResourceEstimate("Kyber spec round-3 core-SVP", 32.2, None,
                      "core-SVP", "kyberspec", 0.60, True),
    ResourceEstimate("NIST PQC forum, memory-adjusted", 143.0 * LOG10_2, None,
                      "core-SVP", "kyberofficialcomment", 0.40, True),
]
KYBER768 = [
    ResourceEstimate("Kyber spec round-3 core-SVP", 49.7, None,
                      "core-SVP", "kyberspec", 0.60, True),
    ResourceEstimate("Memory-adjusted (scaled)", 49.7 + (143.0 * LOG10_2 - 32.2),
                      None, "core-SVP", "kyberofficialcomment", 0.40, True),
]
KYBER1024 = [
    ResourceEstimate("Kyber spec round-3 core-SVP", 69.8, None,
                      "core-SVP", "kyberspec", 0.60, True),
    ResourceEstimate("Memory-adjusted (scaled)", 69.8 + (143.0 * LOG10_2 - 32.2),
                      None, "core-SVP", "kyberofficialcomment", 0.40, True),
]

COST_DISTRIBUTIONS = {
    "RSA-2048": RSA2048,
    "ECC-P256": ECCP256,
    "Kyber-512": KYBER512,
    "Kyber-768": KYBER768,
    "Kyber-1024": KYBER1024,
}

# Point values retained from the deterministic model, so the revision can
# report old-vs-new side by side rather than silently replacing numbers.
POINT_COSTS = {"RSA-2048": 9.81, "ECC-P256": 7.90,
                "Kyber-512": 32.2, "Kyber-768": 49.7, "Kyber-1024": 69.8}


def support(alg):
    ests = COST_DISTRIBUTIONS[alg]
    vals = np.array([e.log10_gates for e in ests])
    w = np.array([e.weight for e in ests], dtype=float)
    return vals, w / w.sum()


def sample_cost(alg, rng, size, jitter=0.0):
    """
    Draw C_alg from its published-estimate mixture.

    `jitter` (log10 units) optionally smears each support point with a
    zero-mean Gaussian, representing within-estimate uncertainty
    (compilation overheads, error-correction assumptions) on top of the
    between-estimate disagreement. jitter=0 gives the pure discrete
    mixture, which is the default reported in the paper.
    """
    vals, p = support(alg)
    idx = rng.choice(len(vals), size=size, p=p)
    out = vals[idx]
    if jitter > 0:
        out = out + rng.normal(0.0, jitter, size=size)
    return out


def expected_cost(alg):
    vals, p = support(alg)
    return float(vals @ p)


def cost_spread(alg):
    vals, _ = support(alg)
    return float(vals.max() - vals.min())


def summary_table():
    rows = []
    for alg, ests in COST_DISTRIBUTIONS.items():
        for e in ests:
            rows.append({
                "algorithm": alg, "estimate": e.label,
                "log10_gates": round(e.log10_gates, 2),
                "log10_qubits": (None if e.log10_qubits is None
                                  else round(e.log10_qubits, 2)),
                "convention": e.convention, "weight": e.weight,
                "exact_primitive": e.exact_primitive, "cite": e.citation,
            })
    return rows


if __name__ == "__main__":
    import json
    print(json.dumps(summary_table(), indent=2))
    print()
    for alg in COST_DISTRIBUTIONS:
        print(f"{alg:12s} E[C]={expected_cost(alg):6.2f}  "
              f"point={POINT_COSTS[alg]:6.2f}  spread={cost_spread(alg):5.2f}")
