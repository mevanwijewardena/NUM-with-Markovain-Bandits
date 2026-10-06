import numpy as np
from scipy.optimize import minimize

def generate_markov_trajectory(P, H, initial_distribution):
    n_states = P.shape[0]
    trajectory = np.zeros(H, dtype=int)

    current_state = np.random.choice(
        n_states,
        p=initial_distribution,
    )

    for t in range(H):
        trajectory[t] = current_state
        current_state = np.random.choice(
            n_states,
            p=P[current_state],
        )

    return trajectory

def get_stationary_mean_and_epsilon(states, P):
    """
    Given the state values and transition matrix of a Markov chain,
    return its stationary mean service and epsilon.

    Parameters
    ----------
    states : array-like
        State values S_i(s).
    P : np.ndarray
        Transition probability matrix.

    Returns
    -------
    stationary_mean : float
        Stationary mean of the state values.
    epsilon : float
        Spectral gap epsilon.
    """

    states = np.asarray(states, dtype=float)
    P = np.asarray(P, dtype=float)

    n = len(states)

    # Stationary distribution pi satisfying pi P = pi
    A = np.vstack((P.T - np.eye(n), np.ones(n)))
    b = np.concatenate((np.zeros(n), [1]))

    pi, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # Stationary mean
    stationary_mean = pi @ states

    # Time-reversed transition matrix
    P_rev = np.diag(1 / pi) @ P.T @ np.diag(pi)

    # Multiplicative symmetrization
    M = P_rev @ P

    # Eigenvalues
    eigenvalues = np.linalg.eigvals(M)
    eigenvalues = np.sort(np.real(eigenvalues))[::-1]

    epsilon = eigenvalues[0] - eigenvalues[1]

    return stationary_mean, epsilon,pi



def get_optimal_p_and_A(S_bar, beta):
    """
    Given stationary mean service rates S_bar and beta,
    compute optimal p* and A* for

        max sum_i log(1 + beta A_i)

    subject to
        A_i <= p_i S_bar_i
        p_i >= 0
        sum_i p_i = 1
    """

    S_bar = np.asarray(S_bar, dtype=float)
    n = len(S_bar)

    # Since utility is increasing, A_i* = p_i* S_bar_i
    def objective(p):
        A = p * S_bar
        return -np.sum(np.log(1 + beta * A))

    constraints = {
        "type": "eq",
        "fun": lambda p: np.sum(p) - 1
    }

    bounds = [(0, 1) for _ in range(n)]

    p0 = np.ones(n) / n

    result = minimize(
        objective,
        p0,
        bounds=bounds,
        constraints=constraints
    )

    p_star = result.x
    A_star = p_star * S_bar

    return p_star, A_star, np.sum(np.log(1 + beta * A_star))

def compute_stat_mean_and_epsilon(n,states_, P):
  S_bar = []
  for k in range(n):
    stationary_mean, epsilon,pi = get_stationary_mean_and_epsilon(states_[k], P[k])
    S_bar.append(stationary_mean)
  return S_bar

def compute_eps_min(n,states_, Ps):
  eps_min = 1
  for k in range(n):
     stationary_mean, epsilon,pi = get_stationary_mean_and_epsilon(states_[k], Ps[k])
     eps_min = min(eps_min,epsilon)
  return eps_min
