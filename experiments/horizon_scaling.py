import math
import numpy as np
from markov_bandits.model import (
    compute_eps_min, compute_stat_mean_and_epsilon, generate_markov_trajectory,
    get_optimal_p_and_A, get_stationary_mean_and_epsilon,
)
from markov_bandits.policies import regen_UCB
from markov_bandits.metrics import compute_metrics_exp_2
from markov_bandits.simulation_plots import plot_horizon_scaling

def horizon_scaling(n, beta,kappa, Hs,c_d,c_v,num_sims,states_,Ps,output_folder):
  stationary_dists = []
  for k in range(n):
    _,_,pi = get_stationary_mean_and_epsilon(states_[k], Ps[k])
    stationary_dists.append(pi)
  S_bar = compute_stat_mean_and_epsilon(n,states_, Ps)
  p_star, A_star,optimal_utility = get_optimal_p_and_A(S_bar, beta)
  epsilon_min =  compute_eps_min(n,states_, Ps)

  departure_utilities = np.zeros((len(Hs),num_sims),dtype=np.float32)
  total_queues = np.zeros((len(Hs),num_sims),dtype=np.float32)
  L = 28*kappa/epsilon_min
  for i in range(len(Hs)):
      H = Hs[i]
      V = c_v*np.sqrt(H)
      d = min(H,math.ceil(np.log(H)/(epsilon_min*c_d)))

      for sim_ in range(num_sims):
        state_sequences = []
        print("H = ",H,"Simulation Number:",sim_)
        for k in range(n):
          state_sequences.append(generate_markov_trajectory(Ps[k], H,stationary_dists[k]))
        A_sequence, D_sequence, Q_sequence = regen_UCB(n,H,V,d,L,beta,states_,state_sequences)
        arrival_utility,departure_utility,total_queue = compute_metrics_exp_2(A_sequence, D_sequence, Q_sequence, beta)
        departure_utilities[i,sim_] = departure_utility
        total_queues[i,sim_] = total_queue
  plot_horizon_scaling(Hs,departure_utilities,total_queues,optimal_utility,output_folder)
