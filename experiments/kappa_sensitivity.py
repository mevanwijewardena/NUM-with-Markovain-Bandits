import math
import numpy as np
from markov_bandits.model import (
    compute_eps_min, compute_stat_mean_and_epsilon, generate_markov_trajectory,
    get_optimal_p_and_A, get_stationary_mean_and_epsilon,
)
from markov_bandits.policies import regen_UCB
from markov_bandits.metrics import compute_metrics
from markov_bandits.simulation_plots import plot_Kappa_sensitivity

def kappa_sensitivity(
    n, beta, H, Kappas, c_d, c_v, num_sims,
    states_, Ps, output_folder
):
    stationary_dists = []
    for k in range(n):
        _, _, pi = get_stationary_mean_and_epsilon(
            states_[k], Ps[k]
        )
        stationary_dists.append(pi)

    S_bar = compute_stat_mean_and_epsilon(n, states_, Ps)
    _, _, optimal_utility = get_optimal_p_and_A(S_bar, beta)

    epsilon_min = compute_eps_min(n, states_, Ps)
    V = c_v * np.sqrt(H)
    d = min(
        H,
        math.ceil(np.log(H) / (epsilon_min * c_d))
    )

    departure_utilities = np.zeros(
        (len(Kappas), num_sims, H), dtype=np.float32
    )
    total_queues = np.zeros(
        (len(Kappas), num_sims, H), dtype=np.float32
    )

    # Generate each simulation's Markov trajectories once.
    # Reuse them across kappa values for a direct comparison.
    state_sequences_sims = []
    for sim_ in range(num_sims):
        state_sequences = []
        for k in range(n):
            state_sequences.append(
                generate_markov_trajectory(
                    Ps[k], H, stationary_dists[k]
                )
            )
        state_sequences_sims.append(state_sequences)

    for i, Kappa in enumerate(Kappas):
        L = 2 * Kappa / epsilon_min

        for sim_ in range(num_sims):
            print(
                "Kappa =", Kappa,
                "L =", L,
                "Simulation Number:", sim_
            )

            A_sequence, D_sequence, Q_sequence = regen_UCB(
                n, H, V, d, L, beta, states_,
                state_sequences_sims[sim_]
            )

            (
                arrival_utility,
                departure_utility,
                total_queue,
                normalized_queue,
            ) = compute_metrics(
                A_sequence, D_sequence, Q_sequence, beta
            )

            departure_utilities[i, sim_] = departure_utility
            total_queues[i, sim_] = total_queue

    return plot_Kappa_sensitivity(
        Kappa_values=Kappas,
        departure_utility_runs=departure_utilities,
        total_queue_runs=total_queues,
        optimal_utility=optimal_utility,
        output_folder=output_folder,
        name="Kappa_Sensitivity",
        highlight_Kappa=0.015,
    )
