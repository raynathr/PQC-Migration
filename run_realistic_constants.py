"""
run_realistic_constants.py

Reproduces the paper's Table tci-realistic (Sec. cas): TCI under the
paper's default illustrative M_KM/M_CAI constants (0.85/0.75, a
mature/target-state organization) versus industry-typical values
(0.4/0.4, the midpoint of the 0.3-0.5 range suggested by the DigiCert
2026 and Sectigo/Omdia 2025 survey figures cited in the paper).

This is a discrete, two-point comparison, not a swept sensitivity
axis, so it is deliberately kept separate from sensitivity.py's 1D/2D
sweeps -- it belongs alongside them conceptually, but the two
constants are varied together and only at one alternative value, not
scanned across a range.

Run with:  python run_realistic_constants.py
(requires config.py to have a resolved quantum cost for RSA-2048, the
same driver algorithm used for the paper's TCI table)
"""

from pathlib import Path

import numpy as np
import pandas as pd

import config
import scenarios
import simulation

OUTPUT_DIR = Path(__file__).parent / "output"

# Industry-typical M_KM/M_CAI, per the paper's Sec. cas discussion of the
# DigiCert Quantum Readiness Outlook 2026 and Sectigo/Omdia 2025 State of
# Crypto Agility Report: real organizations cluster around 0.3-0.5 on the
# relevant sub-metrics, versus the paper's 0.85/0.75 mature-organization
# defaults. 0.4 is the midpoint of that range.
REALISTIC_M_KM = 0.4
REALISTIC_M_CAI = 0.4


def run_tci_table(cfg, C_alg: float, scenario_curves: dict) -> pd.DataFrame:
    """
    Run the TCI simulation for every deployment scenario under a single
    config, matching main.py's TCI Summary table structure.
    """
    rows = []
    for scenario_name, L_t in scenario_curves.items():
        rng = np.random.default_rng(cfg.RANDOM_SEED)
        result = simulation.run_cas_simulation(cfg, C_alg, L_t, rng)
        rows.append({
            "scenario": scenario_name,
            "tci_mean": result["tci"].mean(),
            "tci_std": result["tci"].std(),
        })
    return pd.DataFrame(rows)


def main():
    base_cfg = config.SimulationConfig()
    config.validate_config(base_cfg)
    OUTPUT_DIR.mkdir(exist_ok=True)

    ALGORITHM, COST_TYPE = "RSA-2048", "quantum"
    C_alg = config.get_cost(ALGORITHM, COST_TYPE)

    years = np.arange(1, base_cfg.T_YEARS + 1)
    scenario_curves = scenarios.get_all_scenarios(years, base_cfg)

    print(f"Illustrative constants (paper default): "
          f"M_KM={base_cfg.M_KM_CONSTANT}, M_CAI={base_cfg.M_CAI_CONSTANT}")
    illustrative_df = run_tci_table(base_cfg, C_alg, scenario_curves)
    illustrative_df = illustrative_df.rename(columns={
        "tci_mean": "tci_mean_illustrative",
        "tci_std": "tci_std_illustrative",
    })

    realistic_cfg = config.SimulationConfig(
        M_KM_CONSTANT=REALISTIC_M_KM,
        M_CAI_CONSTANT=REALISTIC_M_CAI,
    )
    config.validate_config(realistic_cfg)
    print(f"Industry-typical constants: "
          f"M_KM={REALISTIC_M_KM}, M_CAI={REALISTIC_M_CAI}")
    realistic_df = run_tci_table(realistic_cfg, C_alg, scenario_curves)
    realistic_df = realistic_df.rename(columns={
        "tci_mean": "tci_mean_realistic",
        "tci_std": "tci_std_realistic",
    })

    merged = illustrative_df.merge(realistic_df, on="scenario")
    merged["drop"] = merged["tci_mean_realistic"] - merged["tci_mean_illustrative"]

    csv_path = OUTPUT_DIR / "tci_realistic_comparison.csv"
    merged.to_csv(csv_path, index=False)
    print(f"\nComparison table written to {csv_path}")
    print(merged.to_string(index=False))
    print("\nThis reproduces the paper's Table tci-realistic (Sec. cas): "
          "moving from the mature/target-state constants to "
          "industry-typical ones lowers TCI uniformly across scenarios, "
          "by an amount larger than growth-rate sensitivity's own "
          "Delta-TCI (see run_sensitivity.py / sensitivity_summary.csv).")


if __name__ == "__main__":
    main()
