"""
run_sensitivity.py

Runs the sensitivity analysis described in paper Section XIII and
produces:
  - output/sensitivity_summary.csv   : tornado-style delta_TCI table
  - output/sensitivity_mu_g.csv      : full mu_g sweep
  - output/sensitivity_alpha.csv     : full alpha sweep
  - output/sensitivity_w_as.csv      : full w_AS sweep
  - output/fig_sensitivity_tornado.png
  - output/fig_sensitivity_heatmap.png

Run with:  python run_sensitivity.py
(requires config.py to have a resolved quantum cost for the algorithm
selected below -- RSA-2048 is used as the driver, consistent with
main.py's case study framing)
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import scenarios
import sensitivity

OUTPUT_DIR = Path(__file__).parent / "output"


def main():
    cfg = config.SimulationConfig()
    config.validate_config(cfg)
    OUTPUT_DIR.mkdir(exist_ok=True)

    ALGORITHM = "RSA-2048"
    COST_TYPE = "quantum"
    C_alg = config.get_cost(ALGORITHM, COST_TYPE)
    print(f"Sensitivity analysis driven by {ALGORITHM} ({COST_TYPE}, C_alg={C_alg})")

    years = np.arange(1, cfg.T_YEARS + 1)
    # Use the Conservative scenario as the reference deployment curve for
    # single-parameter sweeps (it had the highest TCI in the base run,
    # see main.py output) -- this is a modeling choice made explicit here,
    # not hidden.
    L_t = scenarios.conservative_linear(years, cfg.CONSERVATIVE_RATE)

    # ------------------------------------------------------------------
    # 1D sweeps
    # ------------------------------------------------------------------
    mu_g_range = np.linspace(0.15, 0.60, 9)  # E[g] band from paper Sec. growth/sensitivity
    alpha_range = np.linspace(1.0, 3.0, 9)
    w_as_range = np.linspace(0.30, 0.50, 9)

    print("Running mu_g sweep...")
    mu_g_df = sensitivity.scan_mu_g(cfg, C_alg, L_t, mu_g_range, cfg.RANDOM_SEED)

    print("Running alpha sweep...")
    alpha_df = sensitivity.scan_alpha(cfg, C_alg, L_t, alpha_range, cfg.RANDOM_SEED)

    print("Running w_AS sweep...")
    w_as_df = sensitivity.scan_w_as(cfg, C_alg, L_t, w_as_range, cfg.RANDOM_SEED)

    mu_g_df.to_csv(OUTPUT_DIR / "sensitivity_mu_g.csv", index=False)
    alpha_df.to_csv(OUTPUT_DIR / "sensitivity_alpha.csv", index=False)
    w_as_df.to_csv(OUTPUT_DIR / "sensitivity_w_as.csv", index=False)

    summary_df = sensitivity.summarize_sensitivity(mu_g_df, alpha_df, w_as_df)
    summary_df.to_csv(OUTPUT_DIR / "sensitivity_summary.csv", index=False)
    print("\nSensitivity summary (tornado data):")
    print(summary_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Tornado chart
    # ------------------------------------------------------------------
    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=(6, 3.5))
    params = summary_df["parameter"]
    deltas = summary_df["delta_tci"]
    y_pos = np.arange(len(params))
    ax.barh(y_pos, deltas, color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([
        f"{p} [{row.range_min:.2f}, {row.range_max:.2f}]"
        for p, row in zip(params, summary_df.itertuples())
    ])
    ax.set_xlabel(r"$\Delta$TCI across swept range")
    ax.set_title("Sensitivity of TCI to Model Parameters\n(Conservative scenario, RSA-2048 driver)")
    fig.tight_layout()
    tornado_path = OUTPUT_DIR / "fig_sensitivity_tornado.png"
    fig.savefig(tornado_path, dpi=300)
    plt.close(fig)
    print(f"\nTornado chart written to {tornado_path}")

    # ------------------------------------------------------------------
    # 2D heatmap: mu_g x w_AS -> TCI
    # ------------------------------------------------------------------
    print("\nRunning joint (mu_g, w_AS) heatmap grid (this runs multiple simulations)...")
    heatmap_mu_g_range = np.linspace(0.3, 0.7, 7)
    heatmap_w_as_range = np.linspace(0.30, 0.50, 7)
    grid = sensitivity.joint_heatmap(
        cfg, C_alg, L_t, heatmap_mu_g_range, heatmap_w_as_range, cfg.RANDOM_SEED
    )

    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="RdYlGn",
                   extent=[heatmap_mu_g_range.min(), heatmap_mu_g_range.max(),
                           heatmap_w_as_range.min(), heatmap_w_as_range.max()],
                   vmin=0.0, vmax=1.0)
    ax.set_xlabel(r"Growth Rate ($\mu_g$)")
    ax.set_ylabel(r"Algorithm Strength Weight ($w_{AS}$)")
    ax.set_title("Sensitivity Analysis: TCI Across Parameter Space\n(Conservative scenario, RSA-2048 driver)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Trust Continuity Index (TCI)")

    # Annotate each cell with its value, matching the original paper's figure style
    for i, w_as in enumerate(heatmap_w_as_range):
        for j, mu_g in enumerate(heatmap_mu_g_range):
            ax.text(mu_g, w_as, f"{grid[i, j]:.3f}", ha="center", va="center",
                     fontsize=6, color="black")

    fig.tight_layout()
    heatmap_path = OUTPUT_DIR / "fig_sensitivity_heatmap.png"
    fig.savefig(heatmap_path, dpi=300)
    plt.close(fig)
    print(f"Heatmap written to {heatmap_path}")

    print("\nDone. Use sensitivity_summary.csv's delta_tci column directly in")
    print("paper Section XIII prose (e.g. 'For mu_g in [X,Y]: delta_TCI = Z').")


if __name__ == "__main__":
    main()
