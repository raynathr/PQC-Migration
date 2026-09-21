"""
variance_reduction.py

Importance sampling for the rare-event regime (Reviewer 2, major concern
2; Editor comment 7).

The problem, as reported in the previous draft: for Kyber-1024 the mean
RQR is dominated by growth-rate draws far out in the right tail of
Gamma(khat, thetahat). At N = 5000 the estimate rests on a handful of
draws; the estimator was still moving between N = 1e5 and N = 1e6, while
the median sat ~50 orders of magnitude lower. That is a variance problem,
not a sample-size problem, and the previous draft deferred it.

Fix: exponential tilting of the growth-rate distribution. Gamma is closed
under scale changes, so we sample ghat from Gamma(khat, s*thetahat) with
a tilt factor s > 1 and reweight:

    w(g) = f_{k,theta}(g) / f_{k,s*theta}(g)
         = s^k * exp(-g (1/theta - 1/(s theta)))

The estimator mean(w * h(g)) is unbiased for E_f[h(g)] for any s > 0; s
is chosen to put the proposal's mass where the integrand h(g) = RQR(g)
actually lives. We also report Latin Hypercube Sampling of the same
Gamma as a cheaper stratified alternative, so the comparison is
naive MC vs. stratified MC vs. importance sampling rather than a single
technique asserted to help.
"""

import numpy as np


def tilt_weights(g, khat, thetahat, s):
    """Likelihood ratio f_target(g) / f_proposal(g) for a Gamma scale tilt."""
    return (s ** khat) * np.exp(-g * (1.0 / thetahat - 1.0 / (s * thetahat)))


def rqr_of_g(g, years, d_alg, alpha_tilde, eps=None):
    delta = g[:, None] * years[None, :] - d_alg
    if eps is not None:
        delta = delta + eps
    return 1.0 / (1.0 + np.exp(-np.clip(alpha_tilde * delta, -700, 700)))


def naive_mc(rng, n, khat, thetahat, years, d_alg, alpha_tilde, sigma_eps):
    g = rng.gamma(khat, thetahat, size=n)
    eps = rng.normal(0.0, sigma_eps, size=(n, len(years)))
    r = rqr_of_g(g, years, d_alg, alpha_tilde, eps)
    return r.mean(axis=0), r.std(axis=0) / np.sqrt(n)


def lhs_mc(rng, n, khat, thetahat, years, d_alg, alpha_tilde, sigma_eps):
    """Latin Hypercube over the Gamma quantile space (stratified sampling)."""
    from scipy.stats import gamma as gd
    u = (np.arange(n) + rng.random(n)) / n
    rng.shuffle(u)
    g = gd.ppf(u, a=khat, scale=thetahat)
    eps = rng.normal(0.0, sigma_eps, size=(n, len(years)))
    r = rqr_of_g(g, years, d_alg, alpha_tilde, eps)
    return r.mean(axis=0), r.std(axis=0) / np.sqrt(n)


def importance_mc(rng, n, khat, thetahat, years, d_alg, alpha_tilde, sigma_eps, s):
    g = rng.gamma(khat, s * thetahat, size=n)
    w = tilt_weights(g, khat, thetahat, s)
    eps = rng.normal(0.0, sigma_eps, size=(n, len(years)))
    r = rqr_of_g(g, years, d_alg, alpha_tilde, eps)
    wr = w[:, None] * r
    mean = wr.mean(axis=0)
    se = wr.std(axis=0) / np.sqrt(n)
    ess = (w.sum() ** 2) / np.sum(w ** 2)
    return mean, se, ess


def choose_tilt(khat, thetahat, d_alg, T, candidates=None):
    """
    Pick the tilt that centres the proposal on the boundary of the
    contributing region. The integrand only becomes non-negligible once
    ghat*T is within ~1/alpha_tilde of d_alg, i.e. ghat ~ d_alg / T, so a
    proposal with mean near d_alg / T is the natural choice: solve
    khat * s * thetahat = d_alg / T.
    """
    s_star = d_alg / (T * khat * thetahat)
    if candidates is None:
        candidates = [max(1.0, s_star * f) for f in (0.5, 0.75, 1.0, 1.5, 2.0)]
    return s_star, candidates


def importance_estimate(alg, year, cfg, rng, n=200000, stochastic_cost=True,
                         tilt=None):
    """
    Importance-sampled mean RQR for one algorithm at one year, under either
    attack-cost treatment.

    The tilt is applied to the growth rate only, so the likelihood ratio of
    Eq. (tilt) is unchanged when C_alg is also random: the estimator
    mean(w(g) * RQR(g, C)) remains unbiased for any proposal scale, with C
    drawn from its own (untilted) mixture.

    The tilt is targeted at the LOWEST-cost support point, since that is the
    component that dominates the integrand -- a higher-cost component
    contributes even less mass and is served adequately by the same proposal.
    Each (algorithm, year) cell gets its own tilt, because the contributing
    region moves with the horizon: s* = d_min / (year * k * theta).
    """
    import numpy as np
    import costs

    vals, _ = costs.support(alg)
    C_min = float(vals.min())
    d_min = (C_min - cfg.log10_A0) / (cfg.C_rsa_ref - cfg.log10_A0)
    if tilt is None:
        tilt = max(1.0, d_min / (year * cfg.khat * cfg.thetahat))

    g = rng.gamma(cfg.khat, tilt * cfg.thetahat, size=n)
    w = tilt_weights(g, cfg.khat, cfg.thetahat, tilt)

    if stochastic_cost:
        C = costs.sample_cost(alg, rng, size=n)
    else:
        C = np.full(n, costs.POINT_COSTS[alg], dtype=float)
    d = (C - cfg.log10_A0) / (cfg.C_rsa_ref - cfg.log10_A0)

    eps = rng.normal(0.0, cfg.sigma_eps_hat, size=n)
    delta = g * year - d + eps
    r = 1.0 / (1.0 + np.exp(-np.clip(cfg.alpha_tilde * delta, -700, 700)))
    wr = w * r
    ess = (w.sum() ** 2) / np.sum(w ** 2)
    return {"mean": float(wr.mean()), "se": float(wr.std() / np.sqrt(n)),
            "tilt": float(tilt), "ess": float(ess)}
