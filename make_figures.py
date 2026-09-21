"""make_figures.py -- regenerates every figure in the revised manuscript."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gamma as gd

import costs, exposure, metrics, revised_model as rm
import variance_reduction as vr

OUT = Path(__file__).parent / "paper_figures"; OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "axes.grid": True, "grid.alpha": 0.25})
cfg = rm.RevisedConfig()
years = np.arange(1, cfg.T + 1, dtype=float)
rqr, _ = rm.all_rqr(cfg)
SCEN = dict(rm.SCENARIOS); SCEN["Crash"] = lambda t: rm.aggressive(t, 1.0, 1.5, 1.5)

# --- Fig: exposure and coupled vs uncoupled organizational risk -------
fig, ax = plt.subplots(1, 3, figsize=(9.5, 2.9))
for c in exposure.ORG_X:
    ax[0].plot(years, exposure.class_exposure(rm.conservative, c, years),
               label=f"{c.name[:26]} ({c.mode})")
ax[0].set_xlabel("Year"); ax[0].set_ylabel("$E_c(t)$ (legacy-exposed fraction)")
ax[0].set_title("(a) Per-class exposure, Conservative"); ax[0].legend(fontsize=5.5)

for n, f in SCEN.items():
    o = rm.cas_tci(cfg, f, rqr_by_alg=rqr)
    ax[1].plot(years, o["rqr_org"].mean(axis=0), label=n)
ax[1].plot(years, rqr["RSA-2048"].mean(axis=0), "k--", lw=1,
           label="uncoupled (previous model)")
ax[1].set_xlabel("Year"); ax[1].set_ylabel(r"$\mathrm{RQR}_{\mathrm{org}}(t)$")
ax[1].set_title("(b) Migration now moves risk"); ax[1].legend(fontsize=6)

for n, f in SCEN.items():
    o = rm.cas_tci(cfg, f, rqr_by_alg=rqr)
    m = o["cas"].mean(axis=0)
    ax[2].plot(years, m, label=n)
    ax[2].fill_between(years, np.percentile(o["cas"], 2.5, axis=0),
                        np.percentile(o["cas"], 97.5, axis=0), alpha=0.12)
ax[2].axhline(0.725, color="red", ls="--", lw=0.8, label=r"derived $\mathrm{CAS}_{\min}=0.725$")
ax[2].set_xlabel("Year"); ax[2].set_ylabel("CAS(t)")
ax[2].set_title("(c) CAS trajectories"); ax[2].legend(fontsize=6)
fig.tight_layout(); fig.savefig(OUT / "fig_exposure_coupled.png", dpi=300); plt.close(fig)

# --- Fig: attack cost as a distribution -------------------------------
SHORT = {"Gidney 2025": "G25", "Gidney & Ekera 2021": "GE21",
         "Google QAI 2026 (secp256k1 proxy)": "GQ26",
         "Roetteler et al. 2017 (P-256)": "R17",
         "Chevignard et al. 2026 (P-256)": "C26"}
fig, ax = plt.subplots(figsize=(5.2, 2.6))
for alg, y, col in [("RSA-2048", 0, "C0"), ("ECC-P256", 1, "C1")]:
    v, p = costs.support(alg)
    ax.scatter(v, [y] * len(v), s=420 * p, color=col, alpha=0.75, zorder=3,
               edgecolor="k", linewidth=0.4)
    for vi, pi, e in zip(v, p, costs.COST_DISTRIBUTIONS[alg]):
        ax.annotate(f"{SHORT[e.label]}\n{pi:.2f}", (vi, y), textcoords="offset points",
                    xytext=(0, 13), ha="center", fontsize=6.2)
    ax.axvline(costs.expected_cost(alg), ls=":", lw=1.1, color=col, zorder=1)
ax.set_yticks([0, 1]); ax.set_yticklabels(["RSA-2048", "ECC P-256"])
ax.set_ylim(-0.55, 1.75); ax.set_xlim(7.4, 12.0)
ax.set_xlabel(r"$\log_{10}$ Toffoli gates")
ax.set_title("Published quantum attack-cost estimates", fontsize=8.5)
fig.text(0.5, -0.13,
         "GQ26 Google QAI 2026, secp256k1 proxy (min-gate)   |   "
         "R17 Roetteler 2017, P-256 (min-gate)\n"
         "C26 Chevignard 2026, P-256 (min-qubit)   |   G25 Gidney 2025   |   "
         "GE21 Gidney\u2013Eker\u00e5 2021 (min-gate)\n"
         "Marker area $\\propto$ mixture weight $\\pi_e$; dotted lines are mixture means.",
         ha="center", fontsize=5.6)
fig.tight_layout()
fig.savefig(OUT / "fig_cost_distribution.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# --- Fig: variance reduction -----------------------------------------
d = rm.relative_deficit(cfg, costs.POINT_COSTS["Kyber-1024"])
s_star, _ = vr.choose_tilt(cfg.khat, cfg.thetahat, d, cfg.T)
Ns = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000]
res = {"Naive MC": [], "LHS": [], "Importance sampling": []}
for n in Ns:
    r1 = np.random.default_rng(3)
    res["Naive MC"].append(vr.naive_mc(r1, n, cfg.khat, cfg.thetahat, years, d,
                                        cfg.alpha_tilde, cfg.sigma_eps_hat)[0][-1])
    r2 = np.random.default_rng(3)
    res["LHS"].append(vr.lhs_mc(r2, n, cfg.khat, cfg.thetahat, years, d,
                                 cfg.alpha_tilde, cfg.sigma_eps_hat)[0][-1])
    r3 = np.random.default_rng(3)
    res["Importance sampling"].append(
        vr.importance_mc(r3, n, cfg.khat, cfg.thetahat, years, d,
                          cfg.alpha_tilde, cfg.sigma_eps_hat, s_star)[0][-1])
fig, ax = plt.subplots(figsize=(5.2, 3.0))
for k, v in res.items():
    ax.plot(Ns, v, marker="o", ms=3, label=k)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("N"); ax.set_ylabel("Estimated mean RQR, Kyber-1024, Year 15")
ax.set_title(f"Rare-event estimation (tilt $s^*$={s_star:.1f})")
ax.legend(fontsize=6.5)
fig.tight_layout(); fig.savefig(OUT / "fig_variance_reduction.png", dpi=300); plt.close(fig)

# --- Fig: tornado + acceleration --------------------------------------
R = json.load(open(Path(__file__).parent / "output" / "revision_results.json"))
sens = R["sensitivity"]
PRETTY = {
    "kappa_i scaled 0.5x-1.5x": r"$\kappa_i$ (migration coupling)",
    "asset weights v_c +/-0.10": r"asset weights $v_c\pm0.10$",
    "cost weights pi_e": r"cost weights $\pi_e$",
    "sigma_eps in [0.025,0.10]": r"$\sigma_\epsilon\in[0.025,0.10]$",
    "org profile typical vs mature": "org. posture (typical$\\to$mature)",
    "w_AS in [0.30,0.50]": r"$w_{AS}\in[0.30,0.50]$",
    "E[g] in [0.15,0.60]": r"$\mathbb{E}[g]\in[0.15,0.60]$",
    "cost: mixture vs point": "attack cost (mixture vs point)",
    "alpha in [1,3]": r"$\alpha\in[1,3]$",
    "log10 A0 in [2,7] (nuisance)": r"$\log_{10}A_0\in[2,7]$ (nuisance)",
    "hybrid overhead in [0.2,0.8]": r"hybrid overhead $h\in[0.2,0.8]$",
}
items = sorted(sens.items(), key=lambda kv: kv[1]["dTCI"])
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.5))
ax[0].barh([PRETTY.get(k, k) for k, _ in items],
           [v["dTCI"] for _, v in items], color="C0")
for i, (_, v) in enumerate(items):
    ax[0].text(v["dTCI"] + 0.0015, i, f"{v['dTCI']:.4f}".rstrip("0").rstrip("."),
               va="center", fontsize=5.8)
ax[0].set_xlim(0, max(v["dTCI"] for _, v in items) * 1.18)
ax[0].set_xlabel(r"$\Delta$TCI")
ax[0].set_title("(a) Sensitivity of TCI", fontsize=8.5)
ax[0].tick_params(labelsize=6.0)

rows = R["decision_acceleration"]["rows"]
ny = [int(k.split("-")[0]) for k in rows]
l1, = ax[1].plot(ny, [rows[k]["exposure_years"] for k in rows], marker="o",
                 color="C3", label="cumulative exposure-years (left)")
ax[1].set_xlabel("Migration programme duration (years)")
ax[1].set_ylabel("Cumulative exposure-years", color="C3")
ax2 = ax[1].twinx()
l2, = ax2.plot(ny, [rows[k]["tci"] for k in rows], marker="s", color="C0",
               ls="--", label="TCI (right)")
ax2.set_ylabel("TCI", color="C0"); ax2.grid(False)
ax[1].legend(handles=[l1, l2], fontsize=6.5, loc="upper left")
ax[1].set_title("(b) Value of acceleration", fontsize=8.5)
fig.tight_layout(); fig.savefig(OUT / "fig_tornado_decision.png", dpi=300); plt.close(fig)

# --- Fig: convergence, revised model ----------------------------------
rows = []
for n in [100, 200, 500, 1000, 2000, 5000, 10000, 20000]:
    c = rm.RevisedConfig(N=n)
    r, _ = rm.all_rqr(c)
    o = rm.cas_tci(c, rm.conservative, rqr_by_alg=r)
    rows.append((n, o["tci"].mean(), o["rqr_org"][:, -1].mean(),
                 np.percentile(o["rqr_org"][:, -1], 95)))
rows = np.array(rows)
# Two panels sized for a single IEEE column. The 95th-percentile panel is
# dropped: it is constant at 0.1244 for every N (an arithmetic property of
# the asset mix, explained in the text), so plotting it wastes a panel on a
# flat line with a degenerate offset axis.
fig, ax = plt.subplots(1, 2, figsize=(3.45, 1.85))
for i, (lab, col) in enumerate([("mean TCI", 1),
                                 (r"mean $\mathrm{RQR}_{\rm org}(15)$", 2)]):
    ax[i].plot(rows[:, 0], rows[:, col], marker="o", ms=2.5, lw=1.0)
    ax[i].set_xscale("log"); ax[i].set_xlabel("N", fontsize=6)
    ax[i].set_ylabel(lab, fontsize=5.6)
    ax[i].axvline(5000, color="red", ls="--", lw=0.7)
    ax[i].tick_params(labelsize=5.2)
    ax[i].ticklabel_format(axis="y", style="plain", useOffset=False)
fig.tight_layout(pad=0.4)
fig.savefig(OUT / "fig_convergence_revised.png", dpi=400)
plt.close(fig)
np.savetxt(Path(__file__).parent / "output" / "convergence_revised.csv", rows,
           delimiter=",", header="N,tci_mean,rqr_org_y15_mean,rqr_org_y15_p95", comments="")
print("convergence (revised):")
for r_ in rows:
    print("   N=%6d  TCI=%.4f  meanRQRorg15=%.5f  p95=%.5f" % (r_[0], r_[1], r_[2], r_[3]))
print("figures written to", OUT)

# =====================================================================
# Figures added in the revision pass: per-algorithm RQR under the
# revised model, asset-class risk decomposition, the baseline
# comparison, and the 2D sensitivity surface.
# =====================================================================
import json as _json
import costs as _costs
from matplotlib.ticker import LogLocator

rqr_pt, _ = rm.all_rqr(rm.RevisedConfig(stochastic_cost=False))

# --- Fig: per-algorithm RQR evolution, point cost vs cost mixture -----
# Mean (solid) and median (dashed) with the interquartile band. The 95%
# band is uninformative here -- it saturates at [0, 1] for RSA and ECC
# from Year 4 onward -- whereas the mean/median separation is exactly the
# right-skew this model produces and is worth showing.
fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.2), sharey=True)
for i, (src, lab) in enumerate([(rqr_pt, "(a) point attack cost"),
                                 (rqr, "(b) attack-cost mixture")]):
    for j, alg in enumerate(["RSA-2048", "ECC-P256", "Kyber-512"]):
        ax[i].plot(years, src[alg].mean(axis=0), color=f"C{j}", lw=1.6, label=f"{alg} mean")
        ax[i].plot(years, np.median(src[alg], axis=0), color=f"C{j}", lw=1.1,
                   ls="--", label=f"{alg} median")
        ax[i].fill_between(years, np.percentile(src[alg], 25, axis=0),
                            np.percentile(src[alg], 75, axis=0),
                            color=f"C{j}", alpha=0.16, lw=0)
    ax[i].axhline(0.5, color="gray", ls=":", lw=0.8)
    ax[i].set_xlabel("Year"); ax[i].set_title(lab, fontsize=8.5)
ax[0].set_ylabel(r"$\mathrm{RQR}_{\mathrm{alg}}(t)$")
ax[0].set_ylim(-0.02, 0.72)
ax[0].legend(fontsize=5.8, loc="upper left", ncol=2, framealpha=0.92)
fig.suptitle("Per-algorithm break probability: mean (solid), median (dashed), "
             "interquartile band ($N=5000$)", fontsize=8.5)
fig.tight_layout(); fig.savefig(OUT / "fig_rqr_evolution.png", dpi=300); plt.close(fig)

# --- Fig: where the residual risk actually is -------------------------
contrib, labels = [], []
for c in exposure.ORG_X:
    E = exposure.class_exposure(rm.conservative, c, years)
    contrib.append(c.value_weight * (E * rqr[c.legacy_alg].mean(axis=0)
                                      + (1 - E) * rqr[c.pqc_alg].mean(axis=0)))
    labels.append(f"{c.name} ({c.mode})")
contrib = np.array(contrib)
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.0),
                        gridspec_kw={"width_ratios": [1.0, 1.15]})
ax[0].stackplot(years, contrib, labels=labels, alpha=0.85)
ax[0].set_xlabel("Year"); ax[0].set_ylabel(r"contribution to $\mathrm{RQR}_{\mathrm{org}}(t)$")
ax[0].set_title("(a) Risk decomposition, Conservative", fontsize=8.5)
ax[0].set_ylim(0, contrib.sum(axis=0).max() * 1.42)
ax[0].legend(fontsize=5.6, loc="upper left", framealpha=0.92)
share = contrib[:, -1] / contrib[:, -1].sum()
order = np.argsort(-share)
ax[1].barh([labels[i].split(" (")[0] for i in order][::-1],
           share[order][::-1] * 100, color="C3", alpha=0.85)
ax[1].set_xlabel("share of Year-15 residual risk (%)")
ax[1].set_title("(b) 94% of what remains is one class", fontsize=8.5)
ax[1].tick_params(axis="y", labelsize=6.2)
ax[1].set_xlim(0, 104)
for i, v in enumerate(share[order][::-1] * 100):
    ax[1].text(v + 1.5, i, f"{v:.1f}%", va="center", fontsize=6)
fig.tight_layout(); fig.savefig(OUT / "fig_asset_contributions.png", dpi=300); plt.close(fig)

# --- Fig: baseline comparison ----------------------------------------
ABBR = {"External TLS / VPN termination": "TLS/VPN",
        "Archived data and backups": "Archives",
        "Code signing and firmware": "Code signing",
        "Internal PKI and identity federation": "Internal PKI",
        "Machine-to-machine and embedded": "M2M/embedded"}
R2 = _json.load(open(Path(__file__).parent / "output" / "revision_results.json"))
rows_b = R2["baselines"]["rows"]
short = [ABBR[r["class"]] for r in rows_b]
x = np.arange(len(short)); w = 0.38
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.0))
b3 = [r["B3_det_risk"] for r in rows_b]; b5 = [r["B5_stoch_risk"] for r in rows_b]
ax[0].bar(x - w / 2, b3, w, label="B3 deterministic")
ax[0].bar(x + w / 2, b5, w, label="B5 stochastic")
for xi, a_, b_ in zip(x, b3, b5):
    _r = b_ / a_
    ax[0].annotate(f"{_r:.0f}x" if _r >= 10 else f"{_r:.1f}x",
                    (xi, max(a_, b_) * 1.5), ha="center", fontsize=6)
ax[0].set_yscale("log"); ax[0].set_xticks(x)
ax[0].set_xticklabels(short, rotation=18, ha="right", fontsize=6.5)
ax[0].set_ylim(3e-5, 2.0)
ax[0].set_ylabel(r"horizon-mean class risk $\bar R$")
ax[0].set_title(r"(a) Deterministic vs.\ stochastic, ratio annotated", fontsize=8.5)
ax[0].legend(fontsize=6.5, loc="upper right")
pairs = R2["baselines"]["ranking_stability"]
pk = list(pairs.keys()); pv = [pairs[k] for k in pk]
plab = []
for k in pk:
    a_, b_ = k.split(" > ")
    plab.append(f"{ABBR[a_]} $>$ {ABBR[b_]}")
ax[1].barh(range(len(pk)), pv, color="C2", alpha=0.85)
for i, v in enumerate(pv):
    ax[1].text(v + 0.015, i, f"{v:.3f}", va="center", fontsize=6.5)
ax[1].set_yticks(range(len(pk))); ax[1].set_yticklabels(plab, fontsize=6.5)
ax[1].axvline(0.5, color="gray", ls="--", lw=0.8)
ax[1].set_xlim(0, 1.13); ax[1].set_xlabel("fraction of futures in which the ordering holds")
ax[1].set_title("(b) Which parts of the ranking are firm", fontsize=8.5)
fig.tight_layout(); fig.savefig(OUT / "fig_baselines.png", dpi=300); plt.close(fig)

# --- Fig: 2D sensitivity surface, growth rate x posture ---------------
D_RSA = rm.RevisedConfig().C_rsa_ref - rm.RevisedConfig().log10_A0
g_grid = np.linspace(0.15, 0.60, 7)
p_grid = np.linspace(0.0, 1.0, 7)
Z = np.zeros((len(p_grid), len(g_grid)))
for i, pv in enumerate(p_grid):
    for j, gv in enumerate(g_grid):
        c = rm.RevisedConfig(thetahat=(gv / D_RSA) / rm.RevisedConfig().khat,
                              org_profile=float(pv))
        r_, _ = rm.all_rqr(c)
        Z[i, j] = rm.cas_tci(c, rm.conservative, rqr_by_alg=r_)["tci"].mean()
fig, ax = plt.subplots(figsize=(5.0, 3.4))
im = ax.imshow(Z, origin="lower", aspect="auto", cmap="viridis",
               extent=[g_grid[0], g_grid[-1], p_grid[0], p_grid[-1]])
cs = ax.contour(g_grid, p_grid, Z, colors="w", linewidths=0.6, levels=7)
ax.clabel(cs, inline=True, fontsize=5.5, fmt="%.2f")
ax.scatter([0.370], [0.0], marker="*", s=90, color="red", zorder=5,
           label="calibrated / typical")
ax.set_xlabel(r"growth-rate mean $\mathbb{E}[g]$")
ax.set_ylabel("organizational posture (0 = typical, 1 = mature)")
ax.set_title("TCI surface: posture gradient exceeds growth gradient", fontsize=8.5)
ax.legend(fontsize=6, loc="upper right", framealpha=0.9)
fig.colorbar(im, ax=ax, label="mean TCI")
fig.tight_layout(); fig.savefig(OUT / "fig_sensitivity_heatmap.png", dpi=300); plt.close(fig)
print("revision figures written")
print("  vertical (posture) range at calibrated E[g]: %.4f"
      % (Z[:, 3].max() - Z[:, 3].min()))
print("  horizontal (growth) range at typical posture: %.4f"
      % (Z[0, :].max() - Z[0, :].min()))
