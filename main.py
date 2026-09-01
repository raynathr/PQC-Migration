"""
main.py

Entry point for the PQC migration risk Monte Carlo simulation.

Run with:  python main.py

Produces (all under ./output/):
  - environment_report.txt   : actual hardware/software specs + runtime,
                                 for the paper's "Computational Environment" section
  - results_summary.csv       : mean RQR at year 5 / year 15 per algorithm,
                                 and TCI (mean + std) per deployment scenario
  - fig_rqr_evolution.png     : RQR over time, mean + 95% band, per algorithm
  - fig_cas_trajectories.png  : CAS over time, mean + band, per scenario
"""

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import scipy

import config
import scenarios
import simulation

OUTPUT_DIR = Path(__file__).parent / "output"


def capture_environment(elapsed_seconds: float) -> dict:
    """
    Capture the ACTUAL machine/software environment this run executed
    on. This is what goes in the paper's Computational Environment
    section -- do not hand-edit those numbers afterward; if you run
    this on a different machine, re-run it and use that output.
    """
    env = {
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown (platform.processor() returned empty; "
                                              "try running on the target machine directly, "
                                              "not inside a restricted container)",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "pandas_version": pd.__version__,
        "matplotlib_version": matplotlib.__version__,
        "elapsed_seconds": round(elapsed_seconds, 4),
    }

    try:
        import psutil
        vm = psutil.virtual_memory()
        env["total_ram_gb"] = round(vm.total / (1024 ** 3), 2)
        env["cpu_count_logical"] = psutil.cpu_count(logical=True)
        env["cpu_count_physical"] = psutil.cpu_count(logical=False)
    except ImportError:
        env["total_ram_gb"] = "psutil not installed -- run `pip install psutil` for this field"
        env["cpu_count_logical"] = "unavailable"
        env["cpu_count_physical"] = "unavailable"

    return env


def write_environment_report(env: dict, path: Path) -> None:
    with open(path, "w") as f:
        f.write("Computational Environment Report\n")
        f.write("=" * 40 + "\n")
        f.write("Generated automatically by main.py -- use these exact\n")
        f.write("values in the paper's Computational Environment section.\n")
        f.write("Do not substitute different hardware specs by hand.\n\n")
        for key, value in env.items():
            f.write(f"{key}: {value}\n")
    # Also dump machine-readable copy
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(env, f, indent=2)


def check_convergence(cfg, C_alg: float, rng_seed: int) -> None:
    """
    Quick eyeball convergence check: print mean RQR at year T for
    increasing N. This is NOT the full convergence-sweep module
    described in the paper (that comes later, with a proper plot and
    stated tolerance) -- this is a cheap sanity check to run before
    trusting the N=1000 default.
    """
    print("\nConvergence check (mean RQR at final year, increasing N):")
    for n in (100, 200, 500, 1000):
        rng = np.random.default_rng(rng_seed)
        temp_cfg = config.SimulationConfig(
            N_ITERATIONS=n,
            T_YEARS=cfg.T_YEARS,
            RANDOM_SEED=rng_seed,
            A0=cfg.A0, MU_G=cfg.MU_G, SIGMA_G=cfg.SIGMA_G,
            SIGMA_EPS=cfg.SIGMA_EPS, ALPHA=cfg.ALPHA,
            W_AS=cfg.W_AS, W_KM=cfg.W_KM, W_DC=cfg.W_DC, W_CAI=cfg.W_CAI,
        )
        result = simulation.run_rqr_simulation(temp_cfg, C_alg, rng)
        mean_final_year = result["rqr"][:, -1].mean()
        print(f"  N={n:5d}  mean RQR(T={cfg.T_YEARS}) = {mean_final_year:.6e}")
    print("  (This is a quick eyeball check only. For the full convergence "
          "sweep with a stated tolerance, wider N range, and supporting "
          "plot -- reported in the paper's Convergence subsection -- run "
          "convergence.py.)\n")


def main():
    start_time = time.perf_counter()

    cfg = config.SimulationConfig()
    config.validate_config(cfg)

    OUTPUT_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(cfg.RANDOM_SEED)

    print(f"Running Monte Carlo simulation: N={cfg.N_ITERATIONS}, T={cfg.T_YEARS}, seed={cfg.RANDOM_SEED}")

    # ------------------------------------------------------------------
    # RQR evolution per algorithm in SIMULATION_TARGETS
    # ------------------------------------------------------------------
    rqr_results = {}
    skipped_targets = []

    for target in config.SIMULATION_TARGETS:
        alg = target["algorithm"]
        cost_type = target["cost_type"]
        try:
            C_alg = config.get_cost(alg, cost_type)
        except config.MissingCostError as e:
            print(f"  SKIPPING {alg} ({cost_type}): {e}")
            skipped_targets.append((alg, cost_type, str(e)))
            continue

        iter_rng = np.random.default_rng(cfg.RANDOM_SEED)  # same seed per target for comparability
        result = simulation.run_rqr_simulation(cfg, C_alg, iter_rng)
        rqr_results[alg] = result
        print(f"  Simulated {alg} (cost_type={cost_type}, C_alg={C_alg})")

    if not rqr_results:
        raise RuntimeError(
            "No algorithms in config.SIMULATION_TARGETS had a resolvable "
            "attack cost. Fill in at least one 'quantum' cost in "
            "config.ATTACK_COSTS before running the simulation."
        )

    # Convergence sanity check on the first successfully-simulated target
    first_alg = next(iter(rqr_results))
    first_cost_type = next(t["cost_type"] for t in config.SIMULATION_TARGETS if t["algorithm"] == first_alg)
    check_convergence(cfg, config.get_cost(first_alg, first_cost_type), cfg.RANDOM_SEED)

    # ------------------------------------------------------------------
    # CAS / TCI per deployment scenario, using the first resolvable
    # algorithm's RQR as the algorithm-strength input (matches the
    # paper's single-organization case study framing: one legacy
    # algorithm, three possible migration schedules)
    # ------------------------------------------------------------------
    years = rqr_results[first_alg]["years"]
    scenario_curves = scenarios.get_all_scenarios(years, cfg)

    cas_results = {}
    for scenario_name, L_t in scenario_curves.items():
        scenario_rng = np.random.default_rng(cfg.RANDOM_SEED)
        C_alg = config.get_cost(first_alg, first_cost_type)
        result = simulation.run_cas_simulation(cfg, C_alg, L_t, scenario_rng)
        cas_results[scenario_name] = result
        print(f"  Simulated CAS/TCI for scenario '{scenario_name}' "
              f"(using {first_alg} as the RQR driver)")

    # ------------------------------------------------------------------
    # Results table
    # ------------------------------------------------------------------
    summary_rows = []
    for alg, result in rqr_results.items():
        rqr = result["rqr"]
        yr_idx_5 = 4 if cfg.T_YEARS >= 5 else cfg.T_YEARS - 1
        yr_idx_final = cfg.T_YEARS - 1
        summary_rows.append({
            "metric": f"RQR mean @ Year 5 ({alg})",
            "value": rqr[:, yr_idx_5].mean(),
        })
        summary_rows.append({
            "metric": f"RQR mean @ Year {cfg.T_YEARS} ({alg})",
            "value": rqr[:, yr_idx_final].mean(),
        })
        summary_rows.append({
            "metric": f"RQR 95th percentile @ Year {cfg.T_YEARS} ({alg})",
            "value": np.percentile(rqr[:, yr_idx_final], 95),
        })

    for scenario_name, result in cas_results.items():
        summary_rows.append({
            "metric": f"TCI mean ({scenario_name})",
            "value": result["tci"].mean(),
        })
        summary_rows.append({
            "metric": f"TCI std ({scenario_name})",
            "value": result["tci"].std(),
        })

    if skipped_targets:
        for alg, cost_type, reason in skipped_targets:
            summary_rows.append({
                "metric": f"SKIPPED: {alg} ({cost_type})",
                "value": "see console log / MissingCostError",
            })

    summary_df = pd.DataFrame(summary_rows)
    csv_path = OUTPUT_DIR / "results_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\nResults table written to {csv_path}")
    print(summary_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Figure A: RQR evolution, mean + 95% band, per algorithm
    # ------------------------------------------------------------------
    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=(6, 4))
    for alg, result in rqr_results.items():
        rqr = result["rqr"]
        yrs = result["years"]
        mean_line = rqr.mean(axis=0)
        lower = np.percentile(rqr, 2.5, axis=0)
        upper = np.percentile(rqr, 97.5, axis=0)
        ax.plot(yrs, mean_line, label=f"{alg} (mean)")
        ax.fill_between(yrs, lower, upper, alpha=0.2, label=f"{alg} (95% band)")
    ax.set_xlabel("Years from present")
    ax.set_ylabel("Residual Quantum Risk (RQR)")
    ax.set_title("RQR Evolution (Monte Carlo, N={})".format(cfg.N_ITERATIONS))
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="RQR = 0.5")
    ax.legend(fontsize=7, loc="best")
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig_a_path = OUTPUT_DIR / "fig_rqr_evolution.png"
    fig.savefig(fig_a_path, dpi=300)
    plt.close(fig)
    print(f"Figure written to {fig_a_path}")

    # ------------------------------------------------------------------
    # Figure B: CAS trajectories, mean + band, per scenario
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    for scenario_name, result in cas_results.items():
        cas = result["cas"]
        yrs = result["years"]
        mean_line = cas.mean(axis=0)
        lower = np.percentile(cas, 2.5, axis=0)
        upper = np.percentile(cas, 97.5, axis=0)
        ax.plot(yrs, mean_line, label=f"{scenario_name} (mean)")
        ax.fill_between(yrs, lower, upper, alpha=0.15)
    ax.axhline(cfg.CAS_MIN, color="red", linestyle="--", linewidth=0.8,
               label=f"CAS_min = {cfg.CAS_MIN}")
    ax.set_xlabel("Years from present")
    ax.set_ylabel("Composite Assurance Score (CAS)")
    ax.set_title("CAS Trajectories by Deployment Scenario")
    ax.legend(fontsize=7, loc="best")
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig_b_path = OUTPUT_DIR / "fig_cas_trajectories.png"
    fig.savefig(fig_b_path, dpi=300)
    plt.close(fig)
    print(f"Figure written to {fig_b_path}")

    # ------------------------------------------------------------------
    # Environment report (written last so elapsed time includes
    # everything above -- this is the honest wall-clock cost of a full run)
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - start_time
    env = capture_environment(elapsed)
    env_path = OUTPUT_DIR / "environment_report.txt"
    write_environment_report(env, env_path)
    print(f"\nEnvironment report written to {env_path}")
    print(f"Total elapsed time: {elapsed:.4f} seconds")

    if skipped_targets:
        print("\nWARNING: the following targets were skipped because their "
              "attack cost has not been sourced yet. Fill in config.py "
              "ATTACK_COSTS before your final run:")
        for alg, cost_type, _ in skipped_targets:
            print(f"  - {alg} ({cost_type})")


if __name__ == "__main__":
    main()
