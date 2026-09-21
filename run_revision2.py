"""
run_revision2.py

Second half of the revised manuscript's numeric results:
variance reduction, baseline comparison, decision experiments,
sensitivity analysis, threshold derivation.
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import gamma as gd

import costs, exposure, metrics, threshold, baselines
import variance_reduction as vr
import revised_model as rm

OUT = Path(__file__).parent / "output"; OUT.mkdir(exist_ok=True)
R = json.load(open(OUT / "revision_results.json"))
W = {"AS": 0.40, "KM": 0.25, "DC": 0.20, "CAI": 0.15}
cfg = rm.RevisedConfig()
years = np.arange(1, cfg.T + 1, dtype=float)
SCEN = dict(rm.SCENARIOS)
SCEN["Crash"] = lambda t: rm.aggressive(t, L_max=1.0, k=1.5, t0=1.5)

# =====================================================================
print("== 6. variance reduction for the rare-event regime")
R["variance_reduction"] = {}
for alg in ("Kyber-1024", "Kyber-768"):
    d = rm.relative_deficit(cfg, costs.POINT_COSTS[alg])
    s_star, cands = vr.choose_tilt(cfg.khat, cfg.thetahat, d, cfg.T)
    rows = []
    for n in (5000, 50000, 500000):
        rng = np.random.default_rng(11)
        m_n, se_n = vr.naive_mc(rng, n, cfg.khat, cfg.thetahat, years, d,
                                 cfg.alpha_tilde, cfg.sigma_eps_hat)
        rng = np.random.default_rng(11)
        m_l, se_l = vr.lhs_mc(rng, n, cfg.khat, cfg.thetahat, years, d,
                               cfg.alpha_tilde, cfg.sigma_eps_hat)
        rng = np.random.default_rng(11)
        m_i, se_i, ess = vr.importance_mc(rng, n, cfg.khat, cfg.thetahat, years, d,
                                           cfg.alpha_tilde, cfg.sigma_eps_hat, s_star)
        rows.append({"N": n,
                      "naive_mean_y15": float(m_n[-1]), "naive_se_y15": float(se_n[-1]),
                      "lhs_mean_y15": float(m_l[-1]), "lhs_se_y15": float(se_l[-1]),
                      "is_mean_y15": float(m_i[-1]), "is_se_y15": float(se_i[-1]),
                      "is_ess": float(ess)})
        print("   %-11s N=%7d naive=%.4g(se %.2g) lhs=%.4g(se %.2g) IS=%.4g(se %.2g) ESS=%.0f"
              % (alg, n, m_n[-1], se_n[-1], m_l[-1], se_l[-1], m_i[-1], se_i[-1], ess))
    ref = rows[-1]["is_mean_y15"]
    R["variance_reduction"][alg] = {
        "d_alg": float(d), "tilt_s": float(s_star), "rows": rows,
        "se_ratio_at_5000": float(rows[0]["naive_se_y15"] / rows[0]["is_se_y15"]),
        "naive_N_for_IS5000_precision": float(
            (rows[0]["naive_se_y15"] / rows[0]["is_se_y15"]) ** 2 * 5000),
        "reference_mean": ref,
    }
    print("      -> IS s*=%.1f ; se ratio at N=5000 = %.1fx ; naive N needed = %.3g"
          % (s_star, R["variance_reduction"][alg]["se_ratio_at_5000"],
             R["variance_reduction"][alg]["naive_N_for_IS5000_precision"]))

# =====================================================================
# Per-cell importance-sampled Kyber estimates for the paper's headline
# per-algorithm table, under BOTH attack-cost treatments. Naive sampling
# does not estimate these cells at all at Year 5 (see the ratio column).
print("== 6b. importance-sampled Kyber cells (paper Table III)")
R["kyber_is_cells"] = {}
print("   %-11s %3s %12s %12s %10s" % ("alg", "yr", "naive", "IS", "IS/naive"))
for alg in ("Kyber-512", "Kyber-768", "Kyber-1024"):
    R["kyber_is_cells"][alg] = {}
    for yr in (5, 15):
        cell = {}
        for stoch, tag in ((False, "point"), (True, "mixture")):
            rng = np.random.default_rng(202)
            est = vr.importance_estimate(alg, yr, cfg, rng, 400000,
                                          stochastic_cost=stoch)
            key = "mean_y%d" % yr
            naive = R["rqr_by_alg"][alg]["point_cost" if not stoch
                                          else "stochastic_cost"][key]
            cell[tag] = {"is_mean": est["mean"], "is_se": est["se"],
                          "tilt": est["tilt"], "ess": est["ess"],
                          "naive_mean": naive,
                          "ratio_is_over_naive": est["mean"] / naive}
            print("   %-11s %3d %12.3g %12.3g %10.3g  (%s)"
                  % (alg, yr, naive, est["mean"], est["mean"] / naive, tag))
        R["kyber_is_cells"][alg][yr] = cell

print("== 7. baseline comparison")
SEC_LIFETIME = {"External TLS / VPN termination": 10.0,
                 "Archived data and backups": 25.0,
                 "Code signing and firmware": 12.0,
                 "Internal PKI and identity federation": 5.0,
                 "Machine-to-machine and embedded": 15.0}
med_g = float(gd.ppf(0.5, a=cfg.khat, scale=cfg.thetahat))
rqr_sto, _ = rm.all_rqr(cfg)
base_L = rm.conservative

def years_to(frac, cls):
    cov = exposure.class_coverage(base_L, cls, np.linspace(0.01, 40, 4000))
    idx = np.argmax(cov >= frac)
    return float(np.linspace(0.01, 40, 4000)[idx]) if cov.max() >= frac else np.inf

rows = []
for c in exposure.ORG_X:
    d = rm.relative_deficit(cfg, costs.expected_cost(c.legacy_alg))
    X = baselines.median_crqc_year(cfg.khat, cfg.thetahat, d)
    Y = years_to(0.90, c)
    Z = SEC_LIFETIME[c.name]
    b1 = "ACT NOW" if baselines.mosca_verdict(X, Y, Z) else "on track"
    cov5 = float(exposure.class_coverage(base_L, c, np.array([5.0]))[0])
    b2 = baselines.readiness_level(cov5)
    E = exposure.class_exposure(base_L, c, years)
    # B3: deterministic RQR, growth fixed at E[ghat], point costs
    Eg = cfg.khat * cfg.thetahat
    def det_rqr(alg):
        dd = rm.relative_deficit(cfg, costs.POINT_COSTS[alg])
        return 1 / (1 + np.exp(-np.clip(cfg.alpha_tilde * (Eg * years - dd), -700, 700)))
    b3 = float((E * det_rqr(c.legacy_alg) + (1 - E) * det_rqr(c.pqc_alg)).mean())
    b5v = float((E * rqr_sto[c.legacy_alg] + (1 - E) * rqr_sto[c.pqc_alg]).mean(axis=1).mean())
    b5sd = float((E * rqr_sto[c.legacy_alg] + (1 - E) * rqr_sto[c.pqc_alg]).mean(axis=1).std())
    rows.append({"class": c.name, "X": X, "Y": Y, "Z": Z, "B1": b1, "B2": b2,
                  "B3_det_risk": b3, "B5_stoch_risk": b5v, "B5_sd": b5sd,
                  "value_weight": c.value_weight,
                  "B5_weighted": c.value_weight * b5v})
    print("   %-34s X=%.1f Y=%.1f Z=%.0f | B1=%-8s B2=%-10s B3=%.3g B5=%.4f±%.4f"
          % (c.name[:34], X, Y, Z, b1, b2, b3, b5v, b5sd))

R["baselines"] = {
    "median_ghat": med_g, "rows": rows,
    "discrimination": {
        "B1_mosca": baselines.discrimination([r["B1"] for r in rows]),
        "B2_readiness": baselines.discrimination([r["B2"] for r in rows]),
        "B3_deterministic_rqr": baselines.discrimination([round(r["B3_det_risk"], 4) for r in rows]),
        "B5_full": baselines.discrimination([round(r["B5_weighted"], 4) for r in rows]),
        "n_classes": len(rows),
    },
}
print("   distinct verdicts over %d asset classes: B1=%d B2=%d B3=%d B5=%d"
      % (len(rows), R["baselines"]["discrimination"]["B1_mosca"],
         R["baselines"]["discrimination"]["B2_readiness"],
         R["baselines"]["discrimination"]["B3_deterministic_rqr"],
         R["baselines"]["discrimination"]["B5_full"]))

# B4: CAS with and without uncertainty, scenario ranking
print("   scenario ranking, deterministic CAS (B4) vs full (B5):")
b4 = {}
for name, f in SCEN.items():
    Eg = cfg.khat * cfg.thetahat
    det = {}
    for alg in rm.ALL_ALGS:
        dd = rm.relative_deficit(cfg, costs.POINT_COSTS[alg])
        det[alg] = (1 / (1 + np.exp(-np.clip(cfg.alpha_tilde * (Eg * years - dd), -700, 700))))[None, :]
    o = rm.cas_tci(cfg, f, rqr_by_alg=det)
    b4[name] = float(o["tci"].mean())
R["baselines"]["B4_deterministic_tci"] = b4
print("     ", {k: round(v, 4) for k, v in b4.items()})
print("      full :", {k: round(v["tci_mean"], 4) for k, v in R["scenarios"].items()})

# =====================================================================
# Ranking stability: what the stochastic model adds over the deterministic
# one is not a different ranking but a statement about how firm the
# ranking is. For every ordered pair of asset classes we report the
# fraction of simulated futures in which the ranking holds.
per_class = {}
for c in exposure.ORG_X:
    E = exposure.class_exposure(base_L, c, years)
    per_class[c.name] = (E * rqr_sto[c.legacy_alg] + (1 - E) * rqr_sto[c.pqc_alg]).mean(axis=1) * c.value_weight
names = [c.name for c in exposure.ORG_X]
order = sorted(names, key=lambda n: -per_class[n].mean())
pair_conf = {}
for i in range(len(order) - 1):
    a_, b_ = order[i], order[i + 1]
    pair_conf[f"{a_} > {b_}"] = float((per_class[a_] > per_class[b_]).mean())
R["baselines"]["ranking_stability"] = pair_conf
R["baselines"]["mean_rank_order"] = order
print("   adjacent-pair ranking confidence (stochastic model only):")
for k, v in pair_conf.items():
    print("      %-64s %.3f" % (k[:64], v))
det_vs_sto = {r["class"]: (r["B5_stoch_risk"] / r["B3_det_risk"] if r["B3_det_risk"] > 0 else float("inf"))
               for r in rows}
R["baselines"]["stoch_over_det_ratio"] = det_vs_sto
print("   B5/B3 ratio (how much the deterministic model understates risk):")
for k, v in sorted(det_vs_sto.items(), key=lambda kv: -kv[1]):
    print("      %-40s %.1fx" % (k[:40], v))

print("== 8. decision experiment A: which asset class first?")
# Equal total migration effort, reallocated. Effort = sum of rate factors,
# held fixed at its Org-X value; each policy doubles one class's rate and
# rescales the others to keep the sum constant.
base_factors = {c.name: c.migration_rate_factor for c in exposure.ORG_X}
total = sum(base_factors.values())

def policy(boost_name, mult=2.0):
    f = dict(base_factors); f[boost_name] *= mult
    scale = total / sum(f.values())
    return {k: v * scale for k, v in f.items()}

def evaluate(factors, base=rm.conservative):
    cls = [exposure.AssetClass(c.name, c.value_weight, c.legacy_alg, c.pqc_alg,
                                c.mode, c.shelf_life, factors[c.name], c.rationale)
           for c in exposure.ORG_X]
    o = rm.cas_tci(cfg, base, classes=cls, rqr_by_alg=rqr_sto)
    return o

decA = {}
o0 = evaluate(base_factors)
decA["no prioritisation"] = {"tci": float(o0["tci"].mean()),
                              "int_risk": float(o0["rqr_org"].mean())}
for c in exposure.ORG_X:
    o = evaluate(policy(c.name))
    decA["prioritise: " + c.name] = {"tci": float(o["tci"].mean()),
                                      "int_risk": float(o["rqr_org"].mean())}
for k, v in sorted(decA.items(), key=lambda kv: -kv[1]["tci"]):
    print("   %-52s TCI=%.4f  integrated RQR_org=%.4f" % (k[:52], v["tci"], v["int_risk"]))
R["decision_priority"] = decA

print("== 9. decision experiment B: is accelerating 5y -> 3y worth it?")
decB = {}
for n_years in (3, 4, 5, 7, 10):
    o = rm.cas_tci(cfg, rm.linear_years(n_years), rqr_by_alg=rqr_sto)
    decB[f"{n_years}-year linear"] = {
        "tci": float(o["tci"].mean()), "tci_sd": float(o["tci"].std()),
        "integrated_rqr_org": float(o["rqr_org"].mean()),
        "exposure_years": float(o["rqr_org"].mean(axis=0).sum()),
    }
    print("   %-16s TCI=%.4f  integrated RQR_org=%.5f  exposure-years=%.4f"
          % (f"{n_years}y linear", o["tci"].mean(), o["rqr_org"].mean(),
             o["rqr_org"].mean(axis=0).sum()))
d53 = decB["5-year linear"]["exposure_years"] - decB["3-year linear"]["exposure_years"]
d57 = decB["7-year linear"]["exposure_years"] - decB["5-year linear"]["exposure_years"]
R["decision_acceleration"] = {"rows": decB,
                               "exposure_years_saved_5to3": d53,
                               "exposure_years_added_5to7": d57,
                               "tci_gain_5to3": decB["3-year linear"]["tci"] - decB["5-year linear"]["tci"]}
print("   5y->3y saves %.4f exposure-years; 5y->7y costs %.4f; dTCI(5->3)=%+.4f"
      % (d53, d57, R["decision_acceleration"]["tci_gain_5to3"]))

# =====================================================================
print("== 10. sensitivity (revised model)")
sens = {}
def tci_mean(**kw):
    c = rm.RevisedConfig(**{**cfg.__dict__, **kw})
    r, _ = rm.all_rqr(c)
    return float(rm.cas_tci(c, rm.conservative, rqr_by_alg=r)["tci"].mean())

# Growth rate. Swept over the SAME un-normalized band the previous draft
# used, E[g] in [0.15, 0.60], converted into normalized units by dividing
# by D_RSA so the comparison with the previous sensitivity table is
# like-for-like. E[g] = 0.370 (calibrated) maps to E[ghat] = 0.0770.
D_RSA = cfg.C_rsa_ref - cfg.log10_A0
grid = np.linspace(0.15, 0.60, 7) / D_RSA
vals = [tci_mean(thetahat=g / cfg.khat) for g in grid]
sens["E[g] in [0.15,0.60]"] = {"range": [min(vals), max(vals)], "dTCI": max(vals) - min(vals)}
# alpha
vals = [tci_mean(alpha_tilde=a * 4.81) for a in (1.0, 2.0, 3.0)]
sens["alpha in [1,3]"] = {"range": [min(vals), max(vals)], "dTCI": max(vals) - min(vals)}
# CAS weights
vals = []
for w_as in (0.30, 0.40, 0.50):
    rest = 1 - w_as; sc = rest / 0.60
    vals.append(tci_mean(w_AS=w_as, w_KM=0.25 * sc, w_DC=0.20 * sc, w_CAI=0.15 * sc))
sens["w_AS in [0.30,0.50]"] = {"range": [min(vals), max(vals)], "dTCI": max(vals) - min(vals)}
# attack cost: deterministic min-gate vs min-qubit vs full mixture
vals = []
for stoch in (True, False):
    vals.append(tci_mean(stochastic_cost=stoch))
sens["cost: mixture vs point"] = {"range": [min(vals), max(vals)], "dTCI": max(vals) - min(vals)}
# A0 nuisance parameter
vals = [tci_mean(log10_A0=a) for a in (2.0, 3.0, 5.0, 7.0)]
sens["log10 A0 in [2,7] (nuisance)"] = {"range": [min(vals), max(vals)], "dTCI": max(vals) - min(vals)}
# hybrid overhead -- now testable
vals = [tci_mean(hybrid_overhead=h) for h in (0.20, 0.50, 0.80)]
sens["hybrid overhead in [0.2,0.8]"] = {"range": [min(vals), max(vals)], "dTCI": max(vals) - min(vals)}
# org profile
vals = [tci_mean(org_profile=p) for p in ("typical", "mature")]
sens["org profile typical vs mature"] = {"range": [min(vals), max(vals)], "dTCI": max(vals) - min(vals)}
# Annual volatility sigma_eps.
vals = [tci_mean(sigma_eps_hat=v / D_RSA) for v in (0.025, 0.05, 0.10)]
sens["sigma_eps in [0.025,0.10]"] = {"range": [min(vals), max(vals)], "dTCI": max(vals) - min(vals)}

# Cost-mixture weights pi_e: three defensible allocations of ECC's mass.
import dataclasses
_orig_ecc = list(costs.COST_DISTRIBUTIONS["ECC-P256"])
def _set_ecc(ws):
    costs.COST_DISTRIBUTIONS["ECC-P256"] = [
        dataclasses.replace(e, weight=w) for e, w in zip(_orig_ecc, ws)]
vals = []
for ws in ([1.0, 0.0, 0.0], [0.35, 0.30, 0.35], [0.0, 0.5, 0.5]):
    _set_ecc(ws); vals.append(tci_mean())
costs.COST_DISTRIBUTIONS["ECC-P256"] = _orig_ecc
sens["cost weights pi_e"] = {"range": [min(vals), max(vals)], "dTCI": max(vals) - min(vals)}

# Asset value weights v_c: shift 0.10 between the archive class and the rest.
def _shift_v(delta):
    others = [c for c in exposure.ORG_X if c.name != "Archived data and backups"]
    cls = [dataclasses.replace(
               c, value_weight=(c.value_weight + delta
                                if c.name == "Archived data and backups"
                                else c.value_weight - delta / len(others)))
           for c in exposure.ORG_X]
    r_, _ = rm.all_rqr(cfg)
    return float(rm.cas_tci(cfg, rm.conservative, classes=cls,
                             rqr_by_alg=r_)["tci"].mean())
vals = [_shift_v(d) for d in (-0.10, 0.0, 0.10)]
sens["asset weights v_c +/-0.10"] = {"range": [min(vals), max(vals)], "dTCI": max(vals) - min(vals)}

# Migration-coupling coefficients kappa_i, scaled 0.5x to 1.5x.
_orig_k = {i.key: i.kappa for i in
           metrics.M_KM_INDICATORS_TYPICAL + metrics.M_CAI_INDICATORS_TYPICAL}
def _scale_kappa(f):
    for lst in (metrics.M_KM_INDICATORS_TYPICAL, metrics.M_CAI_INDICATORS_TYPICAL):
        for i, ind in enumerate(lst):
            lst[i] = dataclasses.replace(ind, kappa=min(1.0, _orig_k[ind.key] * f))
vals = []
for f in (0.5, 1.0, 1.5):
    _scale_kappa(f); vals.append(tci_mean())
_scale_kappa(1.0)
sens["kappa_i scaled 0.5x-1.5x"] = {"range": [min(vals), max(vals)], "dTCI": max(vals) - min(vals)}

for k, v in sorted(sens.items(), key=lambda kv: -kv[1]["dTCI"]):
    print("   %-34s TCI in [%.4f, %.4f]  dTCI=%.4f" % (k, v["range"][0], v["range"][1], v["dTCI"]))
R["sensitivity"] = sens

# =====================================================================
print("== 11. derived CAS_min")
R["cas_min"] = threshold.table(W)
R["cas_min_floors"] = threshold.COMPONENT_FLOORS
for k, v in R["cas_min"].items():
    print("   %-48s %.3f" % (k, v))

json.dump(R, open(OUT / "revision_results.json", "w"), indent=2, default=float)
print("\nwrote", OUT / "revision_results.json")
