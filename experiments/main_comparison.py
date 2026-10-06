import math
import numpy as np
from markov_bandits.model import (
    compute_eps_min, compute_stat_mean_and_epsilon, generate_markov_trajectory,
    get_optimal_p_and_A, get_stationary_mean_and_epsilon,
)
from markov_bandits.policies import (
    belief_based, known_mean_regen, naive_slotwise_UCB, regen_UCB, uniform_random,
)
from markov_bandits.metrics import compute_metrics, compute_mean_and_ci
from markov_bandits.simulation_plots import plot_utilities

def plot_main_body_figs(n, beta, alpha,L,H,c_d,V,num_sims,states_,Ps,output_folder):
  S_bar = compute_stat_mean_and_epsilon(n,states_, Ps)
  p_star, A_star,optimal_utility = get_optimal_p_and_A(S_bar, beta)
  epsilon_min =  compute_eps_min(n,states_, Ps)
  d = min(H,math.ceil(np.log(H)/(epsilon_min*c_d)))
  print("d:",d)
  print("Optimal Probs", p_star)
  stationary_dists = []
  for k in range(n):
     _,_,pi = get_stationary_mean_and_epsilon(states_[k], Ps[k])
     stationary_dists.append(pi)
  shape = (num_sims,H)

  arrival_utility_tot_regen = np.zeros(shape,dtype=np.float32)
  departure_utility_tot_regen = np.zeros(shape,dtype=np.float32)
  total_queue_tot_regen = np.zeros(shape,dtype=np.float32)


  arrival_utility_tot_naive = np.zeros(shape,dtype=np.float32)
  departure_utility_tot_naive = np.zeros(shape,dtype=np.float32)
  total_queue_tot_naive = np.zeros(shape,dtype=np.float32)


  arrival_utility_tot_known_regen = np.zeros(shape,dtype=np.float32)
  departure_utility_tot_known_regen = np.zeros(shape,dtype=np.float32)
  total_queue_tot_known_regen = np.zeros(shape,dtype=np.float32)


  arrival_utility_tot_belief = np.zeros(shape,dtype=np.float32)
  departure_utility_tot_belief = np.zeros(shape,dtype=np.float32)
  total_queue_tot_belief = np.zeros(shape,dtype=np.float32)


  arrival_utility_tot_uniform = np.zeros(shape,dtype=np.float32)
  departure_utility_tot_uniform = np.zeros(shape,dtype=np.float32)
  total_queue_tot_uniform = np.zeros(shape,dtype=np.float32)

  for sim_ in range(num_sims):
    state_sequences = []
    print("Simulation Number:",sim_)
    for k in range(n):
      state_sequences.append(generate_markov_trajectory(Ps[k], H,stationary_dists[k]))
    ##Regenerative UCB
    A_sequence_regen, D_sequence_regen, Q_sequence_regen = regen_UCB(n,H,V,d,L,beta,states_,state_sequences)
    arrival_utility_regen,departure_utility_regen,total_queue_regen,normalized_queue_regen = compute_metrics(A_sequence_regen, D_sequence_regen, Q_sequence_regen, beta)
    arrival_utility_tot_regen[sim_,:] = arrival_utility_regen
    departure_utility_tot_regen[sim_,:] = departure_utility_regen
    total_queue_tot_regen[sim_,:] = total_queue_regen


    ##Naive Slotwise UCB
    A_sequence_naive, D_sequence_naive, Q_sequence_naive = naive_slotwise_UCB(n,H,V,L,beta,states_,state_sequences)
    arrival_utility_naive,departure_utility_naive,total_queue_naive,normalized_queue_naive = compute_metrics(A_sequence_naive, D_sequence_naive, Q_sequence_naive, beta)
    arrival_utility_tot_naive[sim_,:] = arrival_utility_naive
    departure_utility_tot_naive[sim_,:] = departure_utility_naive
    total_queue_tot_naive[sim_,:] = total_queue_naive


    #Known-mean Regenerative
    A_sequence_known_regen, D_sequence_known_regen, Q_sequence_known_regen = known_mean_regen(n,H,V,d,S_bar,beta,states_,state_sequences)
    arrival_utility_known_regen,departure_utility_known_regen,total_queue_known_regen,normalized_queue_known_regen = compute_metrics(A_sequence_known_regen, D_sequence_known_regen, Q_sequence_known_regen, beta)
    arrival_utility_tot_known_regen[sim_,:] =  arrival_utility_known_regen
    departure_utility_tot_known_regen[sim_,:] =  departure_utility_known_regen
    total_queue_tot_known_regen[sim_,:] = total_queue_known_regen


    #Known-transition Belied DPP
    A_sequence_belief, D_sequence_belief, Q_sequence_belief = belief_based(n,H,V,beta,states_,Ps,state_sequences)
    arrival_utility_belief,departure_utility_belief,total_queue_belief,normalized_queue_belief = compute_metrics(A_sequence_belief, D_sequence_belief, Q_sequence_belief, beta)
    arrival_utility_tot_belief[sim_,:] = arrival_utility_belief
    departure_utility_tot_belief[sim_,:] = departure_utility_belief
    total_queue_tot_belief[sim_,:] =  total_queue_belief


    #Uniform
    A_sequence_uniform, D_sequence_uniform, Q_sequence_uniform = uniform_random(n,H,V,beta,states_,state_sequences)
    arrival_utility_uniform,departure_utility_uniform,total_queue_uniform,normalized_queue_uniform = compute_metrics(A_sequence_uniform, D_sequence_uniform, Q_sequence_uniform, beta)
    arrival_utility_tot_uniform[sim_,:] = arrival_utility_uniform
    departure_utility_tot_uniform[sim_,:] =  departure_utility_uniform
    total_queue_tot_uniform[sim_,:] =  total_queue_uniform

  #Regen UCB
  arrival_utility_regen,arrival_utility_std_regen,arrival_utility_ci_regen = compute_mean_and_ci(arrival_utility_tot_regen)
  departure_utility_regen,departure_utility_std_regen,departure_utility_ci_regen = compute_mean_and_ci(departure_utility_tot_regen)
  total_queue_regen,total_queue_std_regen,total_queue_ci_regen = compute_mean_and_ci(total_queue_tot_regen)
  #Naive slotwise UCB
  arrival_utility_naive,arrival_utility_std_naive,arrival_utility_ci_naive = compute_mean_and_ci(arrival_utility_tot_naive)
  departure_utility_naive,departure_utility_std_naive,departure_utility_ci_naive = compute_mean_and_ci(departure_utility_tot_naive)
  total_queue_naive,total_queue_std_naive,total_queue_ci_naive = compute_mean_and_ci(total_queue_tot_naive)
  #Known Regnerative
  arrival_utility_known_regen,arrival_utility_std_known_regen,arrival_utility_ci_known_regen = compute_mean_and_ci(arrival_utility_tot_known_regen)
  departure_utility_known_regen,departure_utility_std_known_regen,departure_utility_ci_known_regen = compute_mean_and_ci(departure_utility_tot_known_regen)
  total_queue_known_regen,total_queue_std_known_regen,total_queue_ci_known_regen = compute_mean_and_ci(total_queue_tot_known_regen)
  #Belief based DPP
  arrival_utility_belief,arrival_utility_std_belief,arrival_utility_ci_belief = compute_mean_and_ci(arrival_utility_tot_belief)
  departure_utility_belief,departure_utility_std_belief,departure_utility_ci_belief = compute_mean_and_ci(departure_utility_tot_belief)
  total_queue_belief,total_queue_std_belief,total_queue_ci_belief = compute_mean_and_ci(total_queue_tot_belief)
  #Uniform
  arrival_utility_uniform,arrival_utility_std_uniform,arrival_utility_ci_uniform = compute_mean_and_ci(arrival_utility_tot_uniform)
  departure_utility_uniform,departure_utility_std_uniform,departure_utility_ci_uniform = compute_mean_and_ci(departure_utility_tot_uniform)
  total_queue_uniform,total_queue_std_uniform,total_queue_ci_uniform = compute_mean_and_ci(total_queue_tot_uniform)
  arrival_utilities = [
    arrival_utility_regen,
    arrival_utility_naive,
    arrival_utility_known_regen,
    arrival_utility_belief,
    arrival_utility_uniform,
  ]
  departure_utilities = [
    departure_utility_regen,
    departure_utility_naive,
    departure_utility_known_regen,
    departure_utility_belief,
    departure_utility_uniform,
  ]

  total_queues = [
    total_queue_regen,
    total_queue_naive,
    total_queue_known_regen,
    total_queue_belief,
    total_queue_uniform,
  ]
  arrival_utilities_ci = [
    arrival_utility_ci_regen,
    arrival_utility_ci_naive,
    arrival_utility_ci_known_regen,
    arrival_utility_ci_belief,
    arrival_utility_ci_uniform,
  ]
  departure_utilities_ci = [
    departure_utility_ci_regen,
    departure_utility_ci_naive,
    departure_utility_ci_known_regen,
    departure_utility_ci_belief,
    departure_utility_ci_uniform,
  ]
  total_queues_ci = [
    total_queue_ci_regen,
    total_queue_ci_naive,
    total_queue_ci_known_regen,
    total_queue_ci_belief,
    total_queue_ci_uniform,
  ]
  labels = ["Regenerative UCB", "Naive slotwise UCB", "Known-mean Regenerative", "Known-transition Belief DPP", "Uniform"]
  plot_utilities(
    arrival_utilities,
    arrival_utilities_ci,
    labels,
    r"$\phi(\bar{\mathbf{A}}(t))$",
    r"Time slot $t$",
    "ArrivalUtility",
    1,
    optimal_utility,
    output_folder,
)
  plot_utilities(
    departure_utilities,
    departure_utilities_ci,
    labels,
    r"$\phi(\bar{\mathbf{D}}(t))$",
    r"Time slot $t$",
    "DepartureUtility",
    1,
    optimal_utility,
    output_folder,
)
  plot_utilities(
    total_queues,
    total_queues_ci,
    labels,
    r"$\sum_{i=1}^{n} Q_i(t)$",
    r"Time slot $t$",
    "TotalQueueSize",
    0,
    optimal_utility,
    output_folder,
)
