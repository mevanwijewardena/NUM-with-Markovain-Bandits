import numpy as np
from markov_bandits.model import get_stationary_mean_and_epsilon

def solve_admission(Q, V, beta):
    """
    Solve the admission-control problem

        min_A  -V * sum_i log(1 + beta*A_i)
               + sum_i Q_i*A_i

    subject to 0 <= A_i <= 1.
    """

    Q = np.asarray(Q, dtype=float)

    A = np.zeros(len(Q))

    for i in range(len(Q)):

        if Q[i] == 0:
            A[i] = 1.0

        else:
            A[i] = np.clip(
                V / Q[i] - 1 / beta,
                0,
                1
            )

    return A

def decisions_(Q, k, service_value, V, beta):

    Q = np.asarray(Q, dtype=float)

    # Admission decision
    A = solve_admission(Q, V, beta)

    # Queue update
    Q_next = Q + A
    D = np.zeros(len(Q))
    D[k] = min(Q_next[k],service_value)
    Q_next[k] -= service_value
    Q_next = np.maximum(Q_next, 0)

    return A, D, Q_next

def update(S_hat, N, k, service_value, phase2, L):
    """
    Paper's UPDATE subroutine.

    Parameters
    ----------
    S_hat : np.ndarray
        Current empirical means.
    N : np.ndarray
        Current Phase-2 sample counts.
    k : int
        Queue scheduled in this slot.
    service_value : float
        Observed S_k(t).
    phase2 : bool
        True if this slot belongs to Phase 2.
    L : float
        UCB confidence parameter.

    Returns
    -------
    S_hat_new : np.ndarray
        Updated empirical means.
    N_new : np.ndarray
        Updated sample counts.
    UCB : np.ndarray
        Updated UCB indices.
    """

    S_hat_new = S_hat.copy()
    N_new = N.copy()

    # Only Phase-2 observations are used
    if phase2:
        old_count = N[k]

        S_hat_new[k] = (
            S_hat[k] * old_count + service_value
        ) / (old_count + 1)

        N_new[k] += 1

    # Total number of Phase-2 observations
    N_total = np.sum(N_new)

    UCB = np.zeros(len(N_new))

    for i in range(len(N_new)):

        if N_new[i] == 0:
            UCB[i] = 1.0

        else:
            confidence = np.sqrt(
                L * np.log(N_total) / N_new[i]
            )

            UCB[i] = min(
                1.0,
                S_hat_new[i] + confidence
            )

    return S_hat_new, N_new, UCB

def run_slot(
    A_sequence,
    D_sequence,
    Q_sequence,
    Q,
    S_hat,
    N,
    k,
    S_t,
    V,
    beta,
    L,
    phase=1,
):
    # Admission, departure, and queue update
    A, D, Q = decisions_(Q, k, S_t, V, beta)

    # Save this slot's values
    A_sequence.append(A.copy())
    D_sequence.append(D.copy())
    Q_sequence.append(Q.copy())

    # UCB update
    S_hat, N, UCB = update(
        S_hat, N, k, S_t, phase, L
    )

    return (
        A_sequence,
        D_sequence,
        Q_sequence,
        Q,
        S_hat,
        N,
        UCB,
    )

## Regenerative UCB
def regen_UCB(n,H,V,d,L,beta,states_,state_sequences):
  t = -1
  A_sequence = []
  D_sequence = []
  Q_sequence = []
  s_values = []
  S_hat = np.zeros(n)
  N = np.zeros(n)
  Q = np.zeros(n)
  for k in range(n):
    t = t+1
    if(t>= H):
      return A_sequence, D_sequence, Q_sequence
    S_t = states_[k][state_sequences[k][t]]
    (A_sequence,D_sequence,Q_sequence,Q,S_hat,N,UCB,) = run_slot(A_sequence,D_sequence,Q_sequence,Q,S_hat,N,k,S_t,V,beta,L,phase=1)
    s_values.append(S_t)
    t = t+1
    if(t>= H):
      return A_sequence, D_sequence, Q_sequence
    S_t = states_[k][state_sequences[k][t]]
    A,D,Q = decisions_(Q, k, S_t, V, beta)
    A_sequence.append(A)
    D_sequence.append(D)
    Q_sequence.append(Q)
    while(S_t != s_values[k]):
      S_hat, N, UCB = update(S_hat, N, k, S_t, 1, L)
      t = t+1
      if(t>= H):
        return A_sequence, D_sequence, Q_sequence
      S_t = states_[k][state_sequences[k][t]]
      A,D,Q = decisions_(Q, k, S_t, V, beta)
      A_sequence.append(A)
      D_sequence.append(D)
      Q_sequence.append(Q)
    S_hat, N, UCB = update(S_hat, N, k, S_t, 0, L)
    if(k == n-1):
       y_k =  np.argmax(Q * UCB)
    for tau in range(d):
      t = t+1
      if(t>= H):
        return A_sequence, D_sequence, Q_sequence
      S_t = states_[k][state_sequences[k][t]]
      (A_sequence,D_sequence,Q_sequence,Q,S_hat,N,UCB,) = run_slot(A_sequence,D_sequence,Q_sequence,Q,S_hat,N,k,S_t,V,beta,L,phase=0)
  while(t<H):
    #print("Here:", N/np.sum(N))
    t = t+1
    if(t>= H):
      return A_sequence, D_sequence, Q_sequence
    S_t = states_[y_k][state_sequences[y_k][t]]
    A,D,Q = decisions_(Q, y_k, S_t, V, beta)
    A_sequence.append(A)
    D_sequence.append(D)
    Q_sequence.append(Q)
    while(S_t != s_values[y_k]):
      S_hat, N, UCB = update(S_hat, N, y_k, S_t, 0, L)
      t = t+1
      if(t>= H):
        return A_sequence, D_sequence, Q_sequence
      S_t = states_[y_k][state_sequences[y_k][t]]
      A,D,Q = decisions_(Q, y_k, S_t, V, beta)
      A_sequence.append(A)
      D_sequence.append(D)
      Q_sequence.append(Q)
    S_hat, N, UCB = update(S_hat, N, y_k, S_t, 1, L)
    t = t+1
    if(t>= H):
      return A_sequence, D_sequence, Q_sequence
    S_t = states_[y_k][state_sequences[y_k][t]]
    A,D,Q = decisions_(Q, y_k, S_t, V, beta)
    A_sequence.append(A)
    D_sequence.append(D)
    Q_sequence.append(Q)
    while(S_t != s_values[y_k]):
      S_hat, N, UCB = update(S_hat, N, y_k, S_t, 1, L)
      t = t+1
      if(t>= H):
        return A_sequence, D_sequence, Q_sequence
      S_t = states_[y_k][state_sequences[y_k][t]]
      A,D,Q = decisions_(Q, y_k, S_t, V, beta)
      A_sequence.append(A)
      D_sequence.append(D)
      Q_sequence.append(Q)
    S_hat, N, UCB = update(S_hat, N, y_k, S_t, 0, L)
    y_k_old = y_k
    y_k =  np.argmax(Q * UCB)
    for tau in range(d):
      t = t+1
      if(t>= H):
        return A_sequence, D_sequence, Q_sequence
      S_t = states_[y_k_old][state_sequences[y_k_old][t]]
      (A_sequence,D_sequence,Q_sequence,Q,S_hat,N,UCB,) = run_slot(A_sequence,D_sequence,Q_sequence,Q,S_hat,N,y_k_old,S_t,V,beta,L,phase=0)
## Known mean regenerative
def known_mean_regen(n,H,V,d,bar_S,beta,states_,state_sequences):
  t = -1
  A_sequence = []
  D_sequence = []
  Q_sequence = []
  s_values = []
  Q = np.zeros(n)
  for k in range(n):
    t = t+1
    if(t>= H):
      return A_sequence, D_sequence, Q_sequence
    S_t = states_[k][state_sequences[k][t]]
    A, D, Q = decisions_(Q, k, S_t, V, beta)
    A_sequence.append(A)
    D_sequence.append(D)
    Q_sequence.append(Q)
    s_values.append(S_t)
    t = t+1
    if(t>= H):
      return A_sequence, D_sequence, Q_sequence
    S_t = states_[k][state_sequences[k][t]]
    A,D,Q = decisions_(Q, k, S_t, V, beta)
    A_sequence.append(A)
    D_sequence.append(D)
    Q_sequence.append(Q)
    while(S_t != s_values[k]):
      t = t+1
      if(t>= H):
        return A_sequence, D_sequence, Q_sequence
      S_t = states_[k][state_sequences[k][t]]
      A,D,Q = decisions_(Q, k, S_t, V, beta)
      A_sequence.append(A)
      D_sequence.append(D)
      Q_sequence.append(Q)
    if(k == n-1):
       y_k =  np.argmax(Q * bar_S)
    for tau in range(d):
      t = t+1
      if(t>= H):
        return A_sequence, D_sequence, Q_sequence
      S_t = states_[k][state_sequences[k][t]]
      A,D,Q = decisions_(Q, k, S_t, V, beta)
      A_sequence.append(A)
      D_sequence.append(D)
      Q_sequence.append(Q)
  while(t<H):
    t = t+1
    if(t>= H):
      return A_sequence, D_sequence, Q_sequence
    S_t = states_[y_k][state_sequences[y_k][t]]
    A,D,Q = decisions_(Q, y_k, S_t, V, beta)
    A_sequence.append(A)
    D_sequence.append(D)
    Q_sequence.append(Q)
    while(S_t != s_values[y_k]):
      t = t+1
      if(t>= H):
        return A_sequence, D_sequence, Q_sequence
      S_t = states_[y_k][state_sequences[y_k][t]]
      A,D,Q = decisions_(Q, y_k, S_t, V, beta)
      A_sequence.append(A)
      D_sequence.append(D)
      Q_sequence.append(Q)
    t = t+1
    if(t>= H):
      return A_sequence, D_sequence, Q_sequence
    S_t = states_[y_k][state_sequences[y_k][t]]
    A,D,Q = decisions_(Q, y_k, S_t, V, beta)
    A_sequence.append(A)
    D_sequence.append(D)
    Q_sequence.append(Q)
    while(S_t != s_values[y_k]):
      t = t+1
      if(t>= H):
        return A_sequence, D_sequence, Q_sequence
      S_t = states_[y_k][state_sequences[y_k][t]]
      A,D,Q = decisions_(Q, y_k, S_t, V, beta)
      A_sequence.append(A)
      D_sequence.append(D)
      Q_sequence.append(Q)
    y_k_old = y_k
    y_k =  np.argmax(Q * bar_S)
    for tau in range(d):
      t = t+1
      if(t>= H):
        return A_sequence, D_sequence, Q_sequence
      S_t = states_[y_k_old][state_sequences[y_k_old][t]]
      A,D,Q = decisions_(Q, y_k_old, S_t, V, beta)
      A_sequence.append(A)
      D_sequence.append(D)
      Q_sequence.append(Q)
##Naive Slot-wise UCB
def naive_slotwise_UCB(n,H,V,L,beta,states_,state_sequences):
  A_sequence = []
  D_sequence = []
  Q_sequence = []
  s_values = []
  S_hat = np.zeros(n)
  N = np.zeros(n)
  Q = np.zeros(n)
  for t in range(min(n,H)):
    S_t = states_[t][state_sequences[t][t]]
    (A_sequence,D_sequence,Q_sequence,Q,S_hat,N,UCB,) = run_slot(A_sequence,D_sequence,Q_sequence,Q,S_hat,N,t,S_t,V,beta,L,phase=1)
  for t in range(n,H):
    y_k =  np.argmax(Q * UCB)
    S_t = states_[y_k][state_sequences[y_k][t]]
    (A_sequence,D_sequence,Q_sequence,Q,S_hat,N,UCB,) = run_slot(A_sequence,D_sequence,Q_sequence,Q,S_hat,N,y_k,S_t,V,beta,L,phase=1)
  return A_sequence, D_sequence, Q_sequence

##Uniform Random
def uniform_random(n,H,V,beta,states_,state_sequences):
  A_sequence = []
  D_sequence = []
  Q_sequence = []
  s_values = []
  Q = np.zeros(n)
  for t in range(H):
    channel = np.random.choice(n)
    S_t = states_[channel][state_sequences[channel][t]]
    A,D,Q = decisions_(Q, channel, S_t, V, beta)
    A_sequence.append(A)
    D_sequence.append(D)
    Q_sequence.append(Q)
  return A_sequence, D_sequence, Q_sequence

##Belief-based
def belief_based(n,H,V,beta,states_,Ps,state_sequences):
  A_sequence = []
  D_sequence = []
  Q_sequence = []
  belief_update = [(-1,-1) for i in range(n)]
  belief_vectors = []
  Q = np.zeros(n)
  for i in range(n):
    stationary_mean, epsilon,pi = get_stationary_mean_and_epsilon(states_[i], Ps[i])
    belief_vectors.append(pi)
  states_np = [np.asarray(states_[i]) for i in range(n)]
  for t in range(H):
    stat_means = np.asarray([belief_vectors[i]@states_np[i] for i in range(n)])
    y_k =  np.argmax(Q * stat_means)
    S_t = states_[y_k][state_sequences[y_k][t]]
    observed_state = state_sequences[y_k][t]
    belief_vectors[y_k] = Ps[y_k][observed_state].copy()
    for j in range(n):
      if(j != y_k):
         belief_vectors[j] = belief_vectors[j] @Ps[j]
    A,D,Q = decisions_(Q, y_k, S_t, V, beta)
    A_sequence.append(A)
    D_sequence.append(D)
    Q_sequence.append(Q)
  return A_sequence, D_sequence, Q_sequence
