import numpy as np

def compute_metrics(A_sequence, D_sequence, Q_sequence, beta):
    A_array = np.asarray(A_sequence, dtype=float)
    D_array = np.asarray(D_sequence, dtype=float)
    Q_array = np.asarray(Q_sequence, dtype=float)

    T = len(A_array)
    time = np.arange(1, T + 1)

    # Cumulative time-average vectors
    A_time_average = np.cumsum(A_array, axis=0) / time[:, None]
    D_time_average = np.cumsum(D_array, axis=0) / time[:, None]

    # Utility of the cumulative time-average vectors
    arrival_utility = np.sum(
        np.log(1 + beta * A_time_average),
        axis=1,
    )

    departure_utility = np.sum(
        np.log(1 + beta * D_time_average),
        axis=1,
    )

    # Total queue backlog at each slot
    total_queue = np.sum(Q_array, axis=1)

    # Terminal backlog divided by elapsed time
    normalized_queue = total_queue / time

    return (
        arrival_utility,
        departure_utility,
        total_queue,
        normalized_queue,
    )

def compute_mean_and_ci(results):
    results = np.asarray(results)

    mean = np.mean(results, axis=0)

    std = np.std(
        results,
        axis=0,
        ddof=1,
    )

    ci = 1.96 * std / np.sqrt(results.shape[0])

    return mean, std, ci

def compute_metrics_exp_2(
    A_sequence,
    D_sequence,
    Q_sequence,
    beta,
):
    A_array = np.asarray(
        A_sequence,
        dtype=float,
    )

    D_array = np.asarray(
        D_sequence,
        dtype=float,
    )

    Q_array = np.asarray(
        Q_sequence,
        dtype=float,
    )

    A_time_average = np.mean(
        A_array,
        axis=0,
    )

    D_time_average = np.mean(
        D_array,
        axis=0,
    )

    arrival_utility = np.sum(
        np.log(
            1 + beta * A_time_average
        )
    )

    departure_utility = np.sum(
        np.log(
            1 + beta * D_time_average
        )
    )

    terminal_total_queue = np.sum(
        Q_array[-1]
    )

    return (
        arrival_utility,
        departure_utility,
        terminal_total_queue,
    )
