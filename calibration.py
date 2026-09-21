"""
calibration.py

Growth-rate calibration and identifiability analysis (paper Sec.
"Growth Rate Parameterization" and "Eliminating the Absolute Capability
Baseline").

Two jobs:

 1. Re-derive the Gamma(k, theta) fit against GRI/evolutionQ crossing-
    probability targets, reproducing the existing config defaults.

 2. Establish the SCALE DEGENERACY that motivates the normalized
    reparameterization introduced in the revision. Reviewer 2 asked us
    either to ground A0 empirically or to reformulate the model so it
    does not depend on an arbitrary absolute starting capability. The
    analysis below shows the second option is available *exactly*,
    because the model is invariant under a joint rescaling of
    (g, D, alpha).

    Model:  Delta(t) = g*t - D_alg + eps_t,   D_alg = C_alg - log10(A0)
            RQR(t)   = sigmoid(alpha * Delta(t))

    For any c > 0, the map (g, D, alpha) -> (c*g, c*D, alpha/c) leaves
    RQR(t) unchanged for every t. Therefore A0 is NOT separately
    identifiable from any observable the model produces -- it is a
    nuisance parameter, not a physical input we failed to source.

    Setting c = 1/D_RSA gives the normalized coordinates used in the
    revision:
            ghat   = g / D_RSA        (fraction of the RSA deficit closed per year)
            d_alg  = D_alg / D_RSA    (relative deficit; d_RSA == 1 by construction)
            atilde = alpha * D_RSA    (logistic sharpness in normalized units)
    in which RSA-2048's entire RQR trajectory is free of A0.
"""

import numpy as np
from scipy.special import gammaincc
from scipy.optimize import brentq, least_squares

# GRI / evolutionQ crossing-probability targets: Pr(CRQC exists within t years).
# 2024 edition midpoints (the paper's calibration targets) plus the
# 2022-vintage 15-year figure used as a held-out check.
TARGETS_2024 = {5: 0.095, 10: 0.265}
TARGETS_2023 = {5: 0.075, 10: 0.240}
HELDOUT_2022_15Y = 0.50


def crossing_prob_gamma(k, theta, D, t):
    """
    Pr(g*t >= D) for g ~ Gamma(k, theta), i.e. the regularized upper
    incomplete gamma Q(k, D/(t*theta)).

    Note the argument depends on D and theta only through D/theta.
    That ratio -- not D and theta separately -- is what any crossing-
    probability target can identify. This is the degeneracy the
    normalized reparameterization exploits.
    """
    return gammaincc(k, D / (t * theta))


def fit_two_target_gamma(targets, D):
    """
    Solve for (k, theta) matching two crossing-probability targets at a
    FIXED D. Reproduces the paper's original calibration when D = 4.81
    (C_RSA = 9.81, log10 A0 = 5).
    """
    (t1, p1), (t2, p2) = sorted(targets.items())

    def theta_for_k(k):
        # invert Q(k, D/(t1*theta)) = p1 for theta
        x = brentq(lambda x: gammaincc(k, x) - p1, 1e-300, 1e8)
        return D / (t1 * x)

    def residual(k):
        return crossing_prob_gamma(k, theta_for_k(k), D, t2) - p2

    k = brentq(residual, 1e-3, 50.0, xtol=1e-12)
    return k, theta_for_k(k)


def demonstrate_scale_degeneracy(k, theta, D, alpha, t_years, scales=(0.5, 1.0, 2.0, 10.0)):
    """
    Numerically confirm that (g, D, alpha) -> (c g, c D, alpha/c) leaves
    the whole RQR trajectory invariant. Returns the max absolute
    deviation in mean RQR across the horizon, over all scale factors.

    Uses a deterministic quadrature over g rather than Monte Carlo so
    the result is exact to quadrature error and cannot be mistaken for
    sampling noise.
    """
    from scipy.stats import gamma as gamma_dist
    # Gauss-Legendre style grid over the g quantile space.
    q = (np.arange(1, 20001) - 0.5) / 20000.0
    g = gamma_dist.ppf(q, a=k, scale=theta)
    years = np.arange(1, t_years + 1)

    base = None
    max_dev = 0.0
    for c in scales:
        delta = (c * g)[:, None] * years[None, :] - c * D
        rqr = 1.0 / (1.0 + np.exp(-np.clip((alpha / c) * delta, -700, 700)))
        mean_rqr = rqr.mean(axis=0)
        if base is None:
            base = mean_rqr
        else:
            max_dev = max(max_dev, np.abs(mean_rqr - base).max())
    return max_dev


def normalized_parameters(k, theta, D_rsa, alpha):
    """
    Convert a fitted (k, theta, D_rsa, alpha) into the A0-free
    normalized coordinates. Gamma is closed under scaling: if
    g ~ Gamma(k, theta) then g/D ~ Gamma(k, theta/D).
    """
    return {
        "khat": k,
        "thetahat": theta / D_rsa,
        "E_ghat": k * theta / D_rsa,
        "sd_ghat": np.sqrt(k) * theta / D_rsa,
        "alpha_tilde": alpha * D_rsa,
    }


def relative_deficit(C_alg, C_rsa, log10_A0):
    """d_alg = (C_alg - log10 A0) / (C_rsa - log10 A0)."""
    return (C_alg - log10_A0) / (C_rsa - log10_A0)


def a0_sensitivity_of_relative_deficit(C_alg, C_rsa, a0_grid):
    """
    How much does the ONE surviving A0 dependence -- the relative
    deficit d_alg of a non-calibration-target algorithm -- move across a
    defensible bracket of log10(A0)? d_RSA is exactly 1 for every A0,
    which is the point: the headline RSA result is A0-free.
    """
    return {a0: relative_deficit(C_alg, C_rsa, a0) for a0 in a0_grid}


def joint_fit_khat_thetahat(targets):
    """
    Fit the NORMALIZED growth distribution ghat ~ Gamma(khat, thetahat)
    directly against crossing-probability targets, using d_RSA = 1.
    No A0 and no absolute D appear anywhere in this fit -- which is the
    whole point of the reparameterization.

        Pr(CRQC by year t) = Pr(ghat * t >= 1) = Q(khat, 1/(t*thetahat))
    """
    items = sorted(targets.items())

    def resid(p):
        khat, thetahat = np.exp(p)  # keep positive
        return [gammaincc(khat, 1.0 / (t * thetahat)) - p_t for t, p_t in items]

    sol = least_squares(resid, x0=np.log([0.7, 0.11]), xtol=1e-15, ftol=1e-15)
    khat, thetahat = np.exp(sol.x)
    return khat, thetahat, sol


def report():
    C_RSA_Q = 9.81
    LOG10_A0_LEGACY = 5.0
    D_RSA_LEGACY = C_RSA_Q - LOG10_A0_LEGACY
    ALPHA_LEGACY = 2.0

    out = {}

    k, theta = fit_two_target_gamma(TARGETS_2024, D_RSA_LEGACY)
    out["legacy_fit"] = {"k": k, "theta": theta, "E_g": k * theta,
                         "sd_g": np.sqrt(k) * theta,
                         "D_rsa": D_RSA_LEGACY}
    out["legacy_check"] = {t: crossing_prob_gamma(k, theta, D_RSA_LEGACY, t)
                            for t in (5, 10, 15)}

    dev = demonstrate_scale_degeneracy(k, theta, D_RSA_LEGACY, ALPHA_LEGACY, 15)
    out["scale_degeneracy_max_dev"] = dev

    out["normalized"] = normalized_parameters(k, theta, D_RSA_LEGACY, ALPHA_LEGACY)

    khat, thetahat, sol = joint_fit_khat_thetahat(TARGETS_2024)
    out["normalized_direct_fit"] = {"khat": khat, "thetahat": thetahat,
                                     "E_ghat": khat * thetahat,
                                     "sd_ghat": np.sqrt(khat) * thetahat,
                                     "residual": float(np.max(np.abs(sol.fun)))}
    out["normalized_crossing"] = {
        t: gammaincc(khat, 1.0 / (t * thetahat)) for t in (5, 10, 15)
    }

    # Residual A0 dependence: only enters through d_alg for algorithms
    # other than the calibration target.
    a0_grid = [2.0, 3.0, 5.0, 7.0]
    out["d_alg_vs_A0"] = {
        alg: a0_sensitivity_of_relative_deficit(c, C_RSA_Q, a0_grid)
        for alg, c in [("ECC-P256", 7.90), ("Kyber-512", 32.2),
                        ("Kyber-768", 49.7), ("Kyber-1024", 69.8),
                        ("RSA-2048", 9.81)]
    }
    return out


if __name__ == "__main__":
    import json
    r = report()
    print(json.dumps(r, indent=2, default=float))
