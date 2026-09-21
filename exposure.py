"""
exposure.py

Exposure-adjusted organizational residual quantum risk (paper Sec.
"Coupling Migration to Residual Risk").

This module closes the structural gap the Editor and both reviewers
identified first: in the previous formulation RQR depended only on
adversarial capability and a fixed algorithm cost, so migrating a system
never removed it from the risk pool, and only M_DC (20% of CAS) differed
between migration scenarios.

The organizational residual risk is now an asset-weighted mixture over
the algorithms an organization is actually exposed through:

    RQR_org(t) = sum_c v_c [ E_c(t) RQR_legacy(c)(t)
                             + (1 - E_c(t)) RQR_pqc(c)(t) ]

with v_c the value/sensitivity weight of asset class c (sum to 1) and
E_c(t) the fraction of that class still exposed through legacy
cryptography at time t. Migrating an asset moves its weight from the
legacy term to the PQC term, which is exactly what was missing.

E_c(t) takes one of two forms, because the two threat modalities in the
paper's own threat model expose assets on different clocks:

  "live"  -- authentication, integrity and signature assets. Exposure is
             instantaneous: E_c(t) = 1 - L_c(t). Once a code-signing key
             is post-quantum, yesterday's signatures are not the problem;
             tomorrow's forgeries are.

  "hndl"  -- confidentiality assets under Harvest Now, Decrypt Later.
             Exposure is retroactive: data transmitted under legacy
             cryptography stays exposed for as long as it stays
             sensitive. With shelf-life S_c,

                 E_c(t) = (1/S_c) * int_{t-S_c}^{t} (1 - L_c(s)) ds,
                 with L_c(s) = 0 for s < 0.

             This is the term that makes migration TIMING matter rather
             than only migration endpoint, and it is where a late start
             is actually punished: coverage reached after the data was
             already transmitted does not retrieve it.
"""

from dataclasses import dataclass
from typing import Callable, List
import numpy as np


@dataclass(frozen=True)
class AssetClass:
    name: str
    value_weight: float      # v_c, sums to 1 across classes
    legacy_alg: str
    pqc_alg: str             # NIST-category proxy, see paper note
    mode: str                # "live" | "hndl"
    shelf_life: float        # S_c in years; used only when mode == "hndl"
    migration_rate_factor: float  # class difficulty multiplier on L(t)
    rationale: str


# ---------------------------------------------------------------------
# Org-X, reconstructed rather than invented: the class mix and the
# relative difficulty factors follow the asset categories in CISA's PQC
# product-categories list and NIST NCCoE SP 1800-38's discovery
# workstream, and the shelf-life figures follow the conventional
# >10-year HNDL sensitivity threshold used in the paper's own playbook.
# ---------------------------------------------------------------------
ORG_X = [
    AssetClass("External TLS / VPN termination", 0.25, "ECC-P256", "Kyber-768",
               "hndl", 10.0, 1.40,
               "Highest migration tractability: terminates at a small number of "
               "managed endpoints with vendor-supplied hybrid support."),
    AssetClass("Archived data and backups", 0.20, "RSA-2048", "Kyber-1024",
               "hndl", 25.0, 0.50,
               "Longest shelf life and the canonical HNDL target; re-encryption "
               "of an archive is slow and storage-bound."),
    AssetClass("Code signing and firmware", 0.15, "RSA-2048", "Kyber-1024",
               "live", 0.0, 0.60,
               "Forgery risk, not harvest risk; migration gated on device and "
               "supply-chain verifier support."),
    AssetClass("Internal PKI and identity federation", 0.20, "RSA-2048", "Kyber-768",
               "live", 0.0, 1.00,
               "Authentication exposure; migration gated on CA and relying-party "
               "software upgrade cycles."),
    AssetClass("Machine-to-machine and embedded", 0.20, "ECC-P256", "Kyber-512",
               "live", 0.0, 0.40,
               "Least tractable: constrained devices, long field lifetimes, "
               "limited remote update capability."),
]


def check_weights(classes: List[AssetClass]):
    s = sum(c.value_weight for c in classes)
    if abs(s - 1.0) > 1e-9:
        raise ValueError(f"asset value weights must sum to 1.0, got {s}")


def class_coverage(base_L: Callable[[np.ndarray], np.ndarray],
                    cls: AssetClass, t: np.ndarray) -> np.ndarray:
    """
    Per-class coverage: the organization-level schedule accelerated or
    retarded by the class difficulty factor, then clipped to [0, 1].
    Time-scaling (rather than amplitude-scaling) is used so that a hard
    class migrates LATER, not to a lower ceiling -- every class does
    eventually complete.
    """
    return np.clip(base_L(t * cls.migration_rate_factor), 0.0, 1.0)


def hndl_exposure(base_L, cls: AssetClass, t: np.ndarray, quad_steps: int = 400):
    """
    E_c(t) = (1/S) int_{t-S}^{t} (1 - L_c(s)) ds, with L_c(s) = 0 for s < 0.

    Evaluated by trapezoidal quadrature on a fixed grid per time point;
    quad_steps=400 gives an absolute error well below 1e-4 for every
    coverage curve used in the paper (verified against quad_steps=4000).
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    out = np.empty_like(t)
    S = cls.shelf_life
    for i, ti in enumerate(t):
        s = np.linspace(ti - S, ti, quad_steps)
        Ls = np.where(s < 0.0, 0.0, class_coverage(base_L, cls, np.maximum(s, 0.0)))
        out[i] = np.trapezoid(1.0 - Ls, s) / S
    return np.clip(out, 0.0, 1.0)


def class_exposure(base_L, cls: AssetClass, t: np.ndarray) -> np.ndarray:
    if cls.mode == "live":
        return 1.0 - class_coverage(base_L, cls, t)
    if cls.mode == "hndl":
        return hndl_exposure(base_L, cls, t)
    raise ValueError(f"unknown exposure mode {cls.mode!r}")


def org_rqr(rqr_by_alg: dict, base_L, classes: List[AssetClass],
            t: np.ndarray) -> np.ndarray:
    """
    Asset-weighted organizational residual risk.

    rqr_by_alg maps algorithm name -> array of shape (n_iter, T) or (T,).
    Returns an array of the same trailing shape.
    """
    check_weights(classes)
    total = None
    for cls in classes:
        E = class_exposure(base_L, cls, t)            # (T,)
        r_leg = np.asarray(rqr_by_alg[cls.legacy_alg])
        r_pqc = np.asarray(rqr_by_alg[cls.pqc_alg])
        contrib = cls.value_weight * (E * r_leg + (1.0 - E) * r_pqc)
        total = contrib if total is None else total + contrib
    return total


def effective_coverage(base_L, classes: List[AssetClass], t: np.ndarray) -> np.ndarray:
    """
    Value-weighted migration coverage, used as M_DC(t). This replaces the
    single organization-wide L(t): an organization that has migrated its
    easy 25% has not migrated 25% of its risk-weighted estate.
    """
    check_weights(classes)
    return sum(c.value_weight * class_coverage(base_L, c, t) for c in classes)


def exposure_report(base_L, classes, t):
    return {c.name: {"coverage": class_coverage(base_L, c, t).tolist(),
                      "exposure": class_exposure(base_L, c, t).tolist()}
            for c in classes}
