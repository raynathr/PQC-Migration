"""
convergence.py

Dedicated Monte Carlo convergence sweep (paper Sec. Computational
Environment / Limitations future work). Supersedes the "eyeball check"
in main.py's check_convergence(), which only went up to N=1000 and
stated no tolerance.

Sweeps N across a wide range, tracks both the mean and the
95th-percentile RQR at the final simulated year (the tail statistic
the eyeball check flagged as the more uncertain one), and reports the
smallest N at which both statistics stay within a stated tolerance of
the N=10000 reference value for all larger N in the sweep.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import simulation

OUTPUT_DIR = Path(__file__).parent / "output"

N_SWEEP = [100, 200, 500, 1000, 2000, 5000, 10000]
TOLERANCE = 0.01  # absolute tolerance on RQR (0-1 scale), paper-stated


def run_convergence_sweep(cfg, C_alg: float, rng_seed: int) -> pd.DataFrame:
    rows = []
    for n in N_SWEEP:
        rng = np.random.default_rng(rng_seed)
        temp_cfg = config.SimulationConfig(
            N_ITERATIONS=n,
            T_YEARS=cfg.T_YEARS,
            RANDOM_SEED=rng_seed,
            A0=cfg.A0,
            GROWTH_MODEL=cfg.GROWTH_MODEL,
            GAMMA_K=cfg.GAMMA_K, GAMMA_THETA=cfg.GAMMA_THETA,
            MU_G=cfg.MU_G, SIGMA_G=cfg.SIGMA_G,
            SIGMA_EPS=cfg.SIGMA_EPS, ALPHA=cfg.ALPHA,
            W_AS=cfg.W_AS, W_KM=cfg.W_KM, W_DC=cfg.W_DC, W_CAI=cfg.W_CAI,
        )
        result = simulation.run_rqr_simulation(temp_cfg, C_alg, rng)
        rqr_final = result["rqr"][:, -1]
        rows.append({
            "N": n,
            "mean_rqr": rqr_final.mean(),
            "p95_rqr": np.percentile(rqr_final, 95),
        })
    return pd.DataFrame(rows)


def find_convergence_point(df: pd.DataFrame, tolerance: float) -> int:
    """
    Smallest N such that both mean_rqr and p95_rqr are within
    `tolerance` of the reference (largest-N) value for that row AND
    every larger N in the sweep (monotonic-from-here convergence, not
    just a single lucky point).
    """
    ref_mean = df["mean_rqr"].iloc[-1]
    ref_p95 = df["p95_rqr"].iloc[-1]
    for i in range(len(df)):
        tail = df.iloc[i:]
        if (tail["mean_rqr"] - ref_mean).abs().max() <= tolerance and \
           (tail["p95_rqr"] - ref_p95).abs().max() <= tolerance:
            return int(df.iloc[i]["N"])
    return int(df["N"].iloc[-1])


def main():
    cfg = config.SimulationConfig()
    config.validate_config(cfg)
    OUTPUT_DIR.mkdir(exist_ok=True)

    ALGORITHM, COST_TYPE = "RSA-2048", "quantum"
    C_alg = config.get_cost(ALGORITHM, COST_TYPE)

    print(f"Convergence sweep: {ALGORITHM} ({COST_TYPE}), "
          f"N in {N_SWEEP}, tolerance={TOLERANCE} (absolute, RQR scale)")
    df = run_convergence_sweep(cfg, C_alg, cfg.RANDOM_SEED)
    df.to_csv(OUTPUT_DIR / "convergence_sweep.csv", index=False)
    print(df.to_string(index=False))

    n_star = find_convergence_point(df, TOLERANCE)
    print(f"\nConverged (within +/-{TOLERANCE} of the N=10000 reference, "
          f"for this N and all larger N tested) at N={n_star}.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    ax1.plot(df["N"], df["mean_rqr"], marker="o")
    ax1.set_xscale("log")
    ax1.set_xlabel("N (Monte Carlo iterations)")
    ax1.set_ylabel("Mean RQR @ Year 15")
    ax1.set_title("Convergence: Mean")
    ax1.axvline(n_star, color="red", linestyle="--", linewidth=0.8)

    ax2.plot(df["N"], df["p95_rqr"], marker="o", color="darkorange")
    ax2.set_xscale("log")
    ax2.set_xlabel("N (Monte Carlo iterations)")
    ax2.set_ylabel("95th-pctile RQR @ Year 15")
    ax2.set_title("Convergence: 95th Percentile")
    ax2.axvline(n_star, color="red", linestyle="--", linewidth=0.8,
                label=f"N={n_star}")
    ax2.legend(fontsize=7)

    fig.suptitle(f"Monte Carlo Convergence, {ALGORITHM} ({COST_TYPE}), "
                  f"Gamma growth model", fontsize=9)
    fig.tight_layout()
    fig_path = OUTPUT_DIR / "fig_convergence.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"Figure written to {fig_path}")

    return df, n_star


if __name__ == "__main__":
    main()
