"""The two simulation scenarios and default settings from the source notebook."""

import numpy as np


BETA = 4
ALPHA = 3.5  # Retained for the notebook's main experiment signature.
C_V = 0.2
C_D = 0.5
L = 0.5
MAIN_HORIZON = 300_000
MAIN_RUNS = 30
SENSITIVITY_HORIZON = 300_000
SENSITIVITY_RUNS = 30
L_VALUES = [0.5, 2, 10, 100, 1_000, 10_000]
HORIZON_VALUES = {
    1: [100_000, 200_000, 300_000, 400_000, 500_000, 600_000],
    2: [3_000, 10_000, 30_000, 100_000, 300_000],
}
HORIZON_RUNS = {1: 5, 2: 20}


def scenario_model(scenario):
    """Return (n, states, transition matrices) for Scenario 1 or 2."""
    if scenario == 1:
        matrices = [
            [[0.97, 0.03], [0.08, 0.92]],
            [[0.94, 0.06], [0.05, 0.95]],
            [[0.985, 0.015], [0.025, 0.975]],
            [[0.9, 0.1], [0.18, 0.82]],
        ]
    elif scenario == 2:
        matrices = [
            [[0.64, 0.36], [0.04, 0.96]],
            [[0.98, 0.02], [0.38, 0.62]],
            [[0.968, 0.032], [0.368, 0.632]],
            [[0.988, 0.012], [0.388, 0.612]],
        ]
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    states = [[0.05, 1] for _ in range(4)]
    return len(states), states, [np.array(matrix) for matrix in matrices]
