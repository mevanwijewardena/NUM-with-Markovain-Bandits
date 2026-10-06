"""Rebuild the paper figures from experiment CSVs under one output root."""

import argparse
from pathlib import Path

from paper_plots.l_sensitivity_2x2 import plot_l_sensitivity_2x2
from paper_plots.main_2x3 import plot_main_2x3


def make_plots(output_dir, figure="all"):
    root = Path(output_dir).expanduser().resolve()
    paper_dir = root / "Paper_Plots"

    if figure in ("all", "main"):
        required = [
            root / f"Scenario_{scenario}" / f"{metric}.csv"
            for scenario in (1, 2)
            for metric in ("ArrivalUtility", "DepartureUtility", "TotalQueueSize")
        ]
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Run the main experiment for both scenarios first. Missing:\n"
                + "\n".join(str(path) for path in missing)
            )
        plot_main_2x3(str(root), output_folder=str(paper_dir))

    if figure in ("all", "l-sensitivity"):
        files = [
            root / f"Scenario_{scenario}" / "L_Sensitivity" / "L_Sensitivity.csv"
            for scenario in (1, 2)
        ]
        missing = [path for path in files if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Run L-sensitivity for both scenarios first. Missing:\n"
                + "\n".join(str(path) for path in missing)
            )
        plot_l_sensitivity_2x2(
            str(files[0]), str(files[1]), output_folder=str(paper_dir)
        )

    print(f"Paper figures saved in: {paper_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", required=True,
        help="The same output root used for the experiments (local or mounted Drive).",
    )
    parser.add_argument(
        "--figure", choices=("all", "main", "l-sensitivity"), default="all"
    )
    args = parser.parse_args()
    make_plots(args.output_dir, args.figure)


if __name__ == "__main__":
    main()
