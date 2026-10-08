"""Run the notebook experiments and save all results under a chosen directory."""

import argparse
from pathlib import Path

import numpy as np

from experiments.horizon_scaling import horizon_scaling
from experiments.kappa_sensitivity import kappa_sensitivity
from experiments.main_comparison import plot_main_body_figs
from experiments.scenarios import (
    BETA, C_D, C_V, HORIZON_C_D, HORIZON_RUNS, HORIZON_VALUES,
    HORIZON_KAPPA, MAIN_KAPPA, KAPPA_VALUES, MAIN_HORIZON, MAIN_RUNS,
    SENSITIVITY_HORIZON, SENSITIVITY_RUNS, scenario_model,
)
from paper_plots.run import make_plots


def comma_separated(text, converter, description):
    try:
        values = [converter(item.strip()) for item in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid {description}: {text}") from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(f"{description} must contain positive values")
    return values


def main():
    parser = argparse.ArgumentParser(
        description="Run the simulations from the notebook under a custom output root."
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Folder for all CSVs and figures; may be a local or mounted Drive path.",
    )
    parser.add_argument(
        "--experiment", choices=("all", "main", "horizon", "kappa-sensitivity"),
        default="all", help="Which experiment to run (default: all).",
    )
    parser.add_argument(
        "--scenario", choices=("1", "2", "both"), default="both",
        help="Which scenario to run (default: both).",
    )
    parser.add_argument(
        "--num-sims", type=int,
        help="Override the notebook's number of independent runs for each experiment.",
    )
    parser.add_argument(
        "--horizon", type=int,
        help="Override H for the main and kappa-sensitivity experiments.",
    )
    parser.add_argument(
        "--horizons", type=lambda s: comma_separated(s, int, "horizons"),
        help="Override horizon-scaling H values, e.g. 60,120.",
    )
    parser.add_argument(
        "--kappa", type=float,
        help="Kappa override (default: 0.025 for main, 0.002 for horizon).",
    )
    parser.add_argument(
        "--kappa-values", type=lambda s: comma_separated(s, float, "kappa values"),
        help="Override the sensitivity grid, e.g. 0.01,0.025,0.1.",
    )
    parser.add_argument(
        "--seed", type=int, help="Seed NumPy randomness for repeatable simulations.",
    )
    args = parser.parse_args()

    if args.num_sims is not None and args.num_sims < 2:
        parser.error("--num-sims must be at least 2 to calculate confidence intervals")
    if args.horizon is not None and args.horizon < 1:
        parser.error("--horizon must be positive")
    if args.kappa is not None and args.kappa <= 0:
        parser.error("--kappa must be positive")

    root = Path(args.output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.seed is not None:
        np.random.seed(args.seed)

    selected = (1, 2) if args.scenario == "both" else (int(args.scenario),)
    experiments = (
        ("main", "horizon", "kappa-sensitivity")
        if args.experiment == "all" else (args.experiment,)
    )

    for scenario in selected:
        n, states, Ps = scenario_model(scenario)
        scenario_dir = root / f"Scenario_{scenario}"
        for experiment in experiments:
            print(f"\nRunning {experiment}, Scenario {scenario}; output: {scenario_dir}", flush=True)
            if experiment == "main":
                H = args.horizon or MAIN_HORIZON
                runs = args.num_sims or MAIN_RUNS
                plot_main_body_figs(
                    n, BETA, args.kappa if args.kappa is not None else MAIN_KAPPA,
                    H, C_D, C_V * np.sqrt(H),
                    runs, states, Ps, str(scenario_dir),
                )
            elif experiment == "horizon":
                horizon_scaling(
                    n, BETA, args.kappa if args.kappa is not None else HORIZON_KAPPA,
                    args.horizons or HORIZON_VALUES[scenario],
                    HORIZON_C_D[scenario], C_V,
                    args.num_sims or HORIZON_RUNS[scenario],
                    states, Ps, str(scenario_dir / "Horizon_Scaling"),
                )
            else:
                kappa_sensitivity(
                    n, BETA, args.horizon or SENSITIVITY_HORIZON,
                    args.kappa_values or KAPPA_VALUES, C_D, C_V,
                    args.num_sims or SENSITIVITY_RUNS, states, Ps,
                    str(scenario_dir / "Kappa_Sensitivity"),
                )

    if args.scenario == "both":
        if "main" in experiments:
            make_plots(root, figure="main")
        if "kappa-sensitivity" in experiments:
            make_plots(root, figure="kappa-sensitivity")
    print(f"\nFinished. All generated files are under: {root}")


if __name__ == "__main__":
    main()
