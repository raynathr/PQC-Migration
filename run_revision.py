"""
run_revision.py

Produces every numeric result reported in the revised manuscript.
Writes output/revision_results.json and prints a readable summary.
"""
import json, time
from pathlib import Path
import numpy as np

import costs, exposure, metrics, threshold, baselines, variance_reduction as vr
import revised_model as rm
from calibration import report as calib_report

OUT = Path(__file__).parent / "output"; OUT.mkdir(exist_ok=True)
R = {}
W = {"AS": 0.40, "KM": 0.25, "DC": 0.20, "CAI": 0.15}

SCEN = dict(rm.SCENARIOS)
# A genuinely front-loaded schedule. The original "Aggressive" logistic
# has its inflection at t0=3 and is BEHIND the linear schedule for the
# first three years; "Crash" (t0=1.5, k=1.5) is what an actually
# front-loaded programme looks like, and is added so the comparison can
# distinguish curve shape from curve speed.
SCEN["Crash"] = lambda t: rm.aggressive(t, L_max=1.0, k=1.5, t0=1.5)

# =====================================================================
print("== 1. calibration / identifiability")
cal = calib_report()
R["calibration"] = {
    "khat": cal["normalized"]["khat"],
    "thetahat": cal["normalized"]["thetahat"],
    "E_ghat": cal["normalized"]["E_ghat"],
    "sd_ghat": cal["normalized"]["sd_ghat"],
    "alpha_tilde": cal["normalized"]["alpha_tilde"],
    "scale_degeneracy_max_dev": cal["scale_degeneracy_max_dev"],
    "crossing_probs": cal["normalized_crossing"],
    "d_alg_vs_A0": cal["d_alg_vs_A0"],
}
print("   ghat ~ Gamma(%.4f, %.5f), E=%.4f sd=%.4f, alpha_tilde=%.2f"
      % (cal["normalized"]["khat"], cal["normalized"]["thetahat"],
         cal["normalized"]["E_ghat"], cal["normalized"]["sd_ghat"],
         cal["normalized"]["alpha_tilde"]))
print("   scale-degeneracy max deviation: %.2e" % cal["scale_degeneracy_max_dev"])

# =====================================================================
print("== 2. probabilistic attack cost")
rng = np.random.default_rng(7)
a = costs.sample_cost("ECC-P256", rng, 400000)
b = costs.sample_cost("RSA-2048", rng, 400000)
R["costs"] = {
    "table": costs.summary_table(),
    "expected": {k: costs.expected_cost(k) for k in costs.COST_DISTRIBUTIONS},
    "point": costs.POINT_COSTS,
    "spread": {k: costs.cost_spread(k) for k in costs.COST_DISTRIBUTIONS},
    "P_ECC_cheaper_than_RSA": float((a < b).mean()),
}
print("   E[C] ECC=%.2f RSA=%.2f ; P(C_ECC < C_RSA)=%.3f"
      % (costs.expected_cost("ECC-P256"), costs.expected_cost("RSA-2048"),
         (a < b).mean()))

# =====================================================================
print("== 3. per-algorithm RQR, deterministic vs stochastic cost")
cfg = rm.RevisedConfig()
cfg_det = rm.RevisedConfig(stochastic_cost=False)
rqr_sto, _ = rm.all_rqr(cfg)
rqr_det, _ = rm.all_rqr(cfg_det)
R["rqr_by_alg"] = {}
for alg in rm.ALL_ALGS:
    s, d = rqr_sto[alg], rqr_det[alg]
    R["rqr_by_alg"][alg] = {
        "stochastic_cost": {"mean_y5": float(s[:, 4].mean()),
                             "mean_y15": float(s[:, 14].mean()),
                             "median_y15": float(np.median(s[:, 14])),
                             "p95_y15": float(np.percentile(s[:, 14], 95))},
        "point_cost": {"mean_y5": float(d[:, 4].mean()),
                        "mean_y15": float(d[:, 14].mean()),
                        "median_y15": float(np.median(d[:, 14])),
                        "p95_y15": float(np.percentile(d[:, 14], 95))},
    }
    print("   %-11s sto y5=%.4g y15=%.4g | pt y5=%.4g y15=%.4g"
          % (alg, s[:, 4].mean(), s[:, 14].mean(), d[:, 4].mean(), d[:, 14].mean()))
# probability ECC is the riskier of the two, per simulated future
p_ecc_riskier = float((rqr_sto["ECC-P256"][:, 14] > rqr_sto["RSA-2048"][:, 14]).mean())
R["rqr_by_alg"]["P_ECC_riskier_than_RSA_y15"] = p_ecc_riskier
print("   P(ECC riskier than RSA at y15) = %.3f" % p_ecc_riskier)

# =====================================================================
print("== 4. exposure-adjusted CAS / TCI by scenario")
R["scenarios"] = {}
for name, f in SCEN.items():
    out = rm.cas_tci(cfg, f, rqr_by_alg=rqr_sto)
    R["scenarios"][name] = {
        "tci_mean": float(out["tci"].mean()), "tci_std": float(out["tci"].std()),
        "rqr_org_y5": float(out["rqr_org"][:, 4].mean()),
        "rqr_org_y15": float(out["rqr_org"][:, 14].mean()),
        "L_eff_y5": float(out["L_eff"][4]), "L_eff_y15": float(out["L_eff"][14]),
        "M_KM_y1": float(out["M_KM"][0]), "M_KM_y15": float(out["M_KM"][14]),
        "M_CAI_y1": float(out["M_CAI"][0]), "M_CAI_y15": float(out["M_CAI"][14]),
        "cas_min_year": int(np.argmin(out["cas"].mean(axis=0)) + 1),
        "cas_mean_min": float(out["cas"].mean(axis=0).min()),
    }
    print("   %-13s TCI=%.4f (sd %.4f)  RQRorg y5=%.4f y15=%.4f"
          % (name, out["tci"].mean(), out["tci"].std(),
             out["rqr_org"][:, 4].mean(), out["rqr_org"][:, 14].mean()))

# Contrast: the OLD, uncoupled formulation on the same inputs.
print("   -- uncoupled (previous formulation) control:")
R["uncoupled_control"] = {}
for name, f in SCEN.items():
    t = np.arange(1, cfg.T + 1, dtype=float)
    L = np.clip(f(t), 0, 1)
    cas = (W["AS"] * (1 - rqr_sto["RSA-2048"]) + W["KM"] * 0.85
           + W["DC"] * L[None, :] + W["CAI"] * 0.75)
    R["uncoupled_control"][name] = {"tci_mean": float(cas.mean(axis=1).mean()),
                                     "tci_std": float(cas.mean(axis=1).std())}
    print("      %-13s TCI=%.4f" % (name, cas.mean(axis=1).mean()))
sp_new = (max(v["tci_mean"] for v in R["scenarios"].values())
          - min(v["tci_mean"] for v in R["scenarios"].values()))
sp_old = (max(v["tci_mean"] for v in R["uncoupled_control"].values())
          - min(v["tci_mean"] for v in R["uncoupled_control"].values()))
R["scenario_spread"] = {"coupled": sp_new, "uncoupled": sp_old}
print("   scenario TCI spread: coupled=%.4f  uncoupled=%.4f" % (sp_new, sp_old))

# =====================================================================
print("== 5. per-asset-class exposure")
t = np.arange(1, cfg.T + 1, dtype=float)
R["asset_classes"] = []
for c in exposure.ORG_X:
    cov = exposure.class_coverage(rm.conservative, c, t)
    E = exposure.class_exposure(rm.conservative, c, t)
    contrib = c.value_weight * (E * rqr_sto[c.legacy_alg].mean(axis=0)
                                 + (1 - E) * rqr_sto[c.pqc_alg].mean(axis=0))
    R["asset_classes"].append({
        "name": c.name, "v": c.value_weight, "legacy": c.legacy_alg,
        "pqc": c.pqc_alg, "mode": c.mode, "shelf_life": c.shelf_life,
        "rate_factor": c.migration_rate_factor,
        "coverage_y5": float(cov[4]), "coverage_y15": float(cov[14]),
        "exposure_y5": float(E[4]), "exposure_y15": float(E[14]),
        "risk_contrib_y15": float(contrib[14]),
        "risk_contrib_integrated": float(contrib.mean()),
    })
    print("   %-34s E@5=%.2f E@15=%.2f contrib@15=%.4f"
          % (c.name[:34], E[4], E[14], contrib[14]))

json.dump(R, open(OUT / "revision_results.json", "w"), indent=2, default=float)
print("\nwrote", OUT / "revision_results.json")
