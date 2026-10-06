# Network utility maximization with restless Markovian bandits

Python simulations for the queueing and scheduling experiments in the accompanying paper. The code is organized from [`notebooks/Makov_Bandits_Simulations.ipynb`](notebooks/Makov_Bandits_Simulations.ipynb). The simulation and plotting functions retain the notebook's implementations; `run.py` supplies the experiment settings and sends all results to a directory you choose.

## What is simulated?

There are four queues. At each time slot, the controller chooses admissions and schedules one queue for service. Each queue's service state evolves as a two-state Markov chain, including while that queue is not scheduled. The state values are `0.05` and `1`; the transition matrices differ between **Scenario 1** and **Scenario 2**. The objective is the utility of the time-average admissions or departures,

\[
\phi(\mathbf{x})=\sum_{i=1}^{4}\log(1+4x_i).
\]

The plots compare **Regenerative UCB** with four reference policies: naive slotwise UCB, known-mean regenerative scheduling, known-transition belief-based drift-plus-penalty, and uniform random scheduling. The horizontal utility reference \(\phi^*\) is the optimum among stationary state-oblivious policies computed from the stationary service means. It is a benchmark, not a claim that state-oblivious policies are globally optimal under Markovian dynamics.

| Experiment | Outputs | Notebook defaults |
| --- | --- | --- |
| Main comparison | Time-average arrival utility, time-average departure utility, and total queue backlog for all five policies; separate scenario CSV/PDF/PNG files plus a combined 2×3 paper figure | \(H=300{,}000\), 30 runs per scenario, \(L=0.5\) |
| \(L\) sensitivity | Regenerative UCB departure utility and backlog for \(L\in\{0.5,2,10,100,1000,10000\}\); separate scenario files plus a combined 2×2 figure | \(H=300{,}000\), 30 runs per scenario |
| Horizon scaling | Regenerative UCB departure-utility shortfall and terminal backlog across horizon values; a figure and raw per-run CSV for each scenario | Scenario 1: six horizons from 100,000 to 600,000 and 5 runs; Scenario 2: 3,000, 10,000, 30,000, 100,000, 300,000 and 20 runs |

Each main-comparison curve is the average over independent runs. The shaded areas in the 2×3 and 2×2 paper figures show pointwise 95% confidence intervals, computed as the sample mean \(\pm 1.96\,s/\sqrt{N}\) from the runs. Thin intervals may be difficult to see. The horizon-scaling experiment uses its own run counts shown above.

## Run everything with one command

Use Python 3.10+ from this repository's root:

```bash
python -m pip install -r requirements.txt
python run.py --output-dir "./results" --seed 2026
```

**Choose any writable directory for `--output-dir`.** The program creates it if needed. An absolute path, a relative path, a path containing spaces, or a mounted Google Drive folder works. `--seed` is optional; specify one if you want repeatable random draws. Without overrides, the command runs all three experiments in both scenarios, then builds both combined paper figures. Full runs are computationally expensive, so use the short check below before committing to the default settings.

### In Google Colab

After the repository files have been uploaded to GitHub, use these Colab cells. You can also upload and extract the repository ZIP in `/content` instead of cloning it.

```python
!git clone https://github.com/mevanwijewardena/Markov-Chain-Bandits.git
%cd /content/Markov-Chain-Bandits
!python -m pip install -r requirements.txt
```

```python
from google.colab import drive
drive.mount("/content/drive")
```

```python
import subprocess

OUTPUT_DIR = "/content/drive/MyDrive/Markov_Chain_Bandits"  # Change this folder.
subprocess.run(
    ["python", "run.py", "--output-dir", OUTPUT_DIR, "--seed", "2026"],
    check=True,
)
```

The output folder is the **root of the results tree**, not the `Scenario_1` folder. The scripts append the scenario and experiment subdirectories themselves. For example, `OUTPUT_DIR = "/content/drive/MyDrive/My Research Results"` saves *all* results under that folder.

### Short end-to-end check

This checks both scenarios, every experiment, both combined figures, the CSV format, and a custom output path without waiting for the paper-sized runs. Use a separate folder so that check results cannot replace real results:

```bash
python run.py \
  --output-dir "./smoke results" \
  --num-sims 2 \
  --horizon 80 \
  --horizons 40,80 \
  --l-values 0.5,2 \
  --seed 123
```

These small runs are a software check, **not** the paper's numerical results.

## Run individual experiments or rebuild figures

Use the **same** output directory for commands whose results should appear in the same paper figure:

```bash
# Run only the main comparison in both scenarios.
python run.py --experiment main --scenario both --output-dir "./results" --seed 2026

# Run L sensitivity in both scenarios.
python run.py --experiment l-sensitivity --scenario both --output-dir "./results" --seed 2026

# Run horizon scaling in Scenario 2 only.
python run.py --experiment horizon --scenario 2 --output-dir "./results" --seed 2026

# Rebuild both combined figures from existing CSVs, without simulating again.
python -m paper_plots.run --figure all --output-dir "./results"
```

`--experiment` accepts `all`, `main`, `horizon`, and `l-sensitivity`; `--scenario` accepts `both`, `1`, and `2`. You can override simulation settings with `--num-sims N` (at least 2), `--horizon H` (main comparison and \(L\) sensitivity), `--horizons H1,H2,...` (horizon scaling), and `--l-values L1,L2,...` (\(L\) sensitivity). `python run.py --help` lists all options. The combined figures require the corresponding experiment CSVs from **both** scenarios; if files are missing, the plot command lists them.

You can also rebuild just one combined figure:

```bash
python -m paper_plots.run --figure main --output-dir "./results"
python -m paper_plots.run --figure l-sensitivity --output-dir "./results"
```

## Where are the files saved?

The layout below is relative to the value of `--output-dir`:

```text
OUTPUT_DIR/
├── Scenario_1/
│   ├── ArrivalUtility.{csv,pdf,png}
│   ├── DepartureUtility.{csv,pdf,png}
│   ├── TotalQueueSize.{csv,pdf,png}
│   ├── Horizon_Scaling/HorizonScaling.{csv,pdf,png}
│   └── L_Sensitivity/L_Sensitivity.{csv,pdf,png}
├── Scenario_2/
│   └── [the same experiment files]
└── Paper_Plots/
    ├── Main_Results_2x3.{pdf,png}
    └── L_Sensitivity_2x2.{pdf,png}
```

The main and \(L\)-sensitivity CSVs include plotted means and lower/upper 95% confidence bounds. The horizon-scaling CSV contains individual run results. Running the same experiment again with the same output directory replaces **that experiment's** files there. In particular, keep the quick check and final paper runs in different directories.

## Repository map

| Path | Purpose |
| --- | --- |
| `run.py` | One entry point for all scenarios and experiments; accepts the custom output root. |
| `experiments/scenarios.py` | The two Markov models and the notebook's original parameter settings. |
| `experiments/main_comparison.py`, `horizon_scaling.py`, `l_sensitivity.py` | Experiment functions copied from the notebook. |
| `markov_bandits/model.py`, `policies.py`, `metrics.py` | Markov models, five scheduling policies, simulation metrics, and the restored confidence-interval helper. |
| `markov_bandits/simulation_plots.py` | Individual experiment plots and CSV writers. |
| `paper_plots/main_2x3.py`, `l_sensitivity_2x2.py`, `run.py` | Combined paper figures and plot-only command. |
| `notebooks/Makov_Bandits_Simulations.ipynb` | The supplied notebook, kept unchanged as a reference. |

For the custom output directory, run the Python scripts rather than the reference notebook: its cells contain their original fixed Google Drive paths. The Scenario 2 experiment in the scripts always writes to `Scenario_2`.
