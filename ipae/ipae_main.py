import math
import numpy as np
from copy import deepcopy
import time
import os

from mdp import MDP
from mdp import value_iteration
from mdp import best_policy
from mdp import policy_iteration

from ipae_mdp import BaseLevelAction
from ipae_mdp import Distribution
from ipae_mdp import IPAEMDP
from ipae_mdp import Process

from ipae_heuristics import evaluate_schedule_known_deadlines
from ipae_heuristics import evaluate_heuristic_known_deadlines
from ipae_heuristics import evaluate_one_heuristic_empirically
from ipae_heuristics import evaluate_all_heuristics_empirically
from ipae_heuristics import compute_optimal_probability_known_deadlines_file_input
from ipae_heuristics import Schedule
from ipae_heuristics import SAE2DynamicProgramming
from ipae_heuristics import RandomHeuristic
from ipae_heuristics import ExecutiveMostPromisingPlanHeuristic
from ipae_heuristics import RoundRobinHeuristic
from ipae_heuristics import ExecutiveRoundRobinHeuristic
from ipae_heuristics import BasicGreedySchemeHeuristic
from ipae_heuristics import ExecutiveBasicGreedySchemeHeuristic
from ipae_heuristics import DDAHeuristic
from ipae_heuristics import ExecutiveDDAHeuristic
from ipae_heuristics import DDAOnMaxSizeHeuristic

# from ipae_input_generator import build_input_from_text_file as old_input_builder
# from ipae_input_generator import create_input

from ipae_new_input_generator import build_input_from_text_file
from ipae_new_input_generator import create_input


def print_states(mdp):
    for s in mdp.states:
        print(s)

def print_transition_function(mdp):
    for s in mdp.states:
        for a in mdp.A:
            str_pairs = "["
            for p in mdp.transitions[s][a]:
                str_pairs += "(" + str(p[0]) + ", " + str(p[1]) + ")" + ", "
            print(str(s) + ", " + str(a) + ": " + str_pairs[:-2] + "]")

def print_policy(mdp,pi):
    for s in mdp.states:
        print(str(s), "->", str(pi[s]))
    
def evaluate_policy(mdp, pi):
    print("probability of success: " + str(mdp.evaluate_policy(pi)))

def measureIPAEMDPRunningtime(folder, inputs, value_iter=True):
    acc_time = 0
    for address in inputs:
        processes, blas, real_tts, real_dls = build_input_from_text_file(folder + "/" + address)
        starting_time = time.time()
        mdp = IPAEMDP(processes, blas)
        if value_iter:
            best_policy(mdp, value_iteration(mdp))
        else:
            policy_iteration(mdp)
        finishing_time = time.time()
        acc_time += finishing_time - starting_time
    func = "value iteration" if value_iter else "policy iteration"
    print("average running time of " + func + " in seconds is:", acc_time / len(inputs))


# input_str = "ipae_input_bank/input1_one_process_no_bl_actions" # 0.5
# input_str = "ipae_input_bank/input2_two_processes_no_bl_actions" # 0.6
# input_str = "ipae_input_bank/input3_one_process_one_bl_action" # 0.5
# input_str = "ipae_input_bank/input4_one_process_two_bl_actions" # 0.5
# input_str = "ipae_input_bank/input5_two_processes_one_bl_action" # 0.84
# input_str = "ipae_input_bank/input6_one_process_no_bl_actions_unknown_deadlines" # 0.35
# input_str = "ipae_input_bank/input7_two_processes_no_bl_actions_unknown_deadlines" # 0.51125
# input_str = "ipae_input_bank/input8_one_process_one_bl_action_unknown_deadlines" # 0.5
# input_str = "ipae_input_bank/input9_one_process_two_bl_actions_unknown_deadlines" # 0.5
# input_str = "ipae_input_bank/input10_two_processes_one_bl_action_unknown_deadlines" # 0.82
# input_str = "ipae_input_bank/input11_three_processes_no_bl_actions_unknown_deadlines" # 0.896
# input_str = "ipae_input_bank/input12_one_process_one_bl_action_known_deadlines" # 0.5
# input_str = "ipae_input_bank/input13_one_process_two_bl_actions_known_deadlines" # 0.5
# input_str = "ipae_input_bank/input14_two_processes_one_bl_action_unknown_deadlines" # 0.85
# input_str = "ipae_input_bank/5_5_kd_20/input0"
# processes, blas, real_tts, real_dls = build_input_from_text_file(input_str)

# mdp = IPAEMDP(processes, blas)
# print_states(mdp)
# print_transition_function(mdp)
# pi = best_policy(mdp, value_iteration(mdp))
# pi = policy_iteration(mdp)
# print_policy(mdp, pi)
# evaluate_policy(mdp, pi)


# schedule = [0,blas[0],0,blas[1],0,0,0]
# print(evaluate_schedule_known_deadlines(processes, schedule))


# heuristic = Schedule([blas[0],0,0,0,blas[1],0,0,0,0])
# heuristic = RandomHeuristic(processes, blas)
# heuristic = ExecutiveMostPromisingPlanHeuristic(processes)
# heuristic = ExecutiveRoundRobinHeuristic(processes)
# heuristic = BasicGreedySchemeHeuristic(processes, 0)
# heuristic = ExecutiveBasicGreedySchemeHeuristic(processes, 0)
# heuristic = DDAHeuristic(processes, 1, 1)
# print(evaluate_heuristic_known_deadlines(processes, blas, heuristic))


# measureIPAEMDPRunningtime("ipae_input_bank", ["input0", "input1"], value_iter=True)


# evaluate_one_heuristic_empirically(["input1_one_process_no_bl_actions"], 10000, "Schedule", [[0,0,0,0,0,0]]) # 0.5
# evaluate_one_heuristic_empirically(["input2_two_processes_no_bl_actions"], 10000, "Schedule", [[0,0,0,0,0,0]]) # 0.5
# evaluate_one_heuristic_empirically(["input2_two_processes_no_bl_actions"], 10000, "Schedule", [[0,1,0,0,0,0]]) # 0.6
# evaluate_one_heuristic_empirically(["input3_one_process_one_bl_action"], 10000, "Schedule", [[0,'0',0,0,0]]) # 0.0
# evaluate_one_heuristic_empirically(["input3_one_process_one_bl_action"], 10000, "Schedule", [['0',0,0,0,0]]) # 0.5
# evaluate_one_heuristic_empirically(["input4_one_process_two_bl_actions"], 10000, "Schedule", [['0',0,0,'1',0,0]]) # 0.2
# evaluate_one_heuristic_empirically(["input4_one_process_two_bl_actions"], 10000, "Schedule", [['0',0,'1',0,0]]) # 0.5
# evaluate_one_heuristic_empirically(["input5_two_processes_one_bl_action"], 10000, "Schedule", [[1,1,1,1,1]]) # 0.8
# evaluate_one_heuristic_empirically(["input5_two_processes_one_bl_action"], 10000, "Schedule", [[1,'0',0,0]]) # 0.84
# evaluate_one_heuristic_empirically(["input5_two_processes_one_bl_action"], 10000, "Schedule", [['0',0,1]]) # 0.2
# evaluate_one_heuristic_empirically(["input5_two_processes_one_bl_action"], 10000, "Schedule", [[1,1,'0',0,0]]) # 0.8
# evaluate_one_heuristic_empirically(["input6_one_process_no_bl_actions_unknown_deadlines"], 10000, "Schedule", [[0,0,0,0,0]]) # 0.35
# evaluate_one_heuristic_empirically(["input7_two_processes_no_bl_actions_unknown_deadlines"], 10000, "Schedule", [[1,1,0]]) # 0.51125
# evaluate_one_heuristic_empirically(["input8_one_process_one_bl_action_unknown_deadlines"], 10000, "Schedule", [['0',0,0,0]]) # 0.5
# evaluate_one_heuristic_empirically(["input8_one_process_one_bl_action_unknown_deadlines"], 10000, "Schedule", [[0,'0',0,0,0]]) # 0.25
# evaluate_one_heuristic_empirically(["input9_one_process_two_bl_actions_unknown_deadlines"], 10000, "Schedule", [['0',0,'1',0,0]]) # 0.5
# evaluate_one_heuristic_empirically(["input9_one_process_two_bl_actions_unknown_deadlines"], 10000, "Schedule", [[0,'0',0,'1',0]]) # 0.1
# evaluate_one_heuristic_empirically(["input10_two_processes_one_bl_action_unknown_deadlines"], 10000, "Schedule", [[1,1,1]]) # 0.8
# evaluate_one_heuristic_empirically(["input10_two_processes_one_bl_action_unknown_deadlines"], 10000, "Schedule", [[1,'0',0]]) # 0.82
# evaluate_one_heuristic_empirically(["input11_three_processes_no_bl_actions_unknown_deadlines"], 10000, "Schedule", [[1,2,0]]) # 0.883
# evaluate_one_heuristic_empirically(["input11_three_processes_no_bl_actions_unknown_deadlines"], 10000, "Schedule", [[1,0,2]]) # 0.896
# evaluate_one_heuristic_empirically(["input12_one_process_one_bl_action_known_deadlines"], 10000, "Schedule", [['0',0,0,0]]) # 0.5
# evaluate_one_heuristic_empirically(["input12_one_process_one_bl_action_known_deadlines"], 10000, "Schedule", [[0,'0',0,0,0]]) # 0.0
# evaluate_one_heuristic_empirically(["input13_one_process_two_bl_actions_known_deadlines"], 10000, "Schedule", [['0',0,'1',0,0,0]]) # 0.5
# evaluate_one_heuristic_empirically(["input13_one_process_two_bl_actions_known_deadlines"], 10000, "Schedule", [[0,'0',0,'1',0,0,0]]) # 0.0
# evaluate_one_heuristic_empirically(["input13_one_process_two_bl_actions_known_deadlines"], 10000, "Schedule", [['0',0,0,'1',0,0,0]]) # 0.5
# evaluate_one_heuristic_empirically(["input13_one_process_two_bl_actions_known_deadlines"], 10000, "Schedule", [['0',0,0,0,'1',0,0]]) # 0.0
# evaluate_one_heuristic_empirically(["input14_two_processes_one_bl_action_unknown_deadlines"], 10000, "Schedule", [[1,1,1,1]]) # 0.85
# evaluate_one_heuristic_empirically(["input14_two_processes_one_bl_action_unknown_deadlines"], 10000, "Schedule", [[1,'0',0,0,0]]) # 0.8

# evaluate_one_heuristic_empirically(["input1_one_process_no_bl_actions"], 10000, "SAE2DynamicProgramming") # 0.5
# evaluate_one_heuristic_empirically(["input2_two_processes_no_bl_actions"], 10000, "SAE2DynamicProgramming") # 0.6
# evaluate_one_heuristic_empirically(["input3_one_process_one_bl_action"], 10000, "SAE2DynamicProgramming") # 0.0
# evaluate_one_heuristic_empirically(["input4_one_process_two_bl_actions"], 10000, "SAE2DynamicProgramming") # 0.0
# evaluate_one_heuristic_empirically(["input5_two_processes_one_bl_action"], 10000, "SAE2DynamicProgramming") # 0.84
# evaluate_one_heuristic_empirically(["input12_one_process_one_bl_action_known_deadlines"], 10000, "SAE2DynamicProgramming") # 0.0
# evaluate_one_heuristic_empirically(["input13_one_process_two_bl_actions_known_deadlines"], 10000, "SAE2DynamicProgramming") # 0.0

# evaluate_one_heuristic_empirically(["input1_one_process_no_bl_actions"], 10000, "Random") # 0.5
# evaluate_one_heuristic_empirically(["input2_two_processes_no_bl_actions"], 10000, "Random") # 0.58
# evaluate_one_heuristic_empirically(["input3_one_process_one_bl_action"], 10000, "Random") # 0.25
# evaluate_one_heuristic_empirically(["input4_one_process_two_bl_actions"], 10000, "Random") # 0.5
# evaluate_one_heuristic_empirically(["input5_two_processes_one_bl_action"], 10000, "Random") # 0.65
# evaluate_one_heuristic_empirically(["input6_one_process_no_bl_actions_unknown_deadlines"], 10000, "Random") # 0.35
# evaluate_one_heuristic_empirically(["input7_two_processes_no_bl_actions_unknown_deadlines"], 10000, "Random") # 0.5
# evaluate_one_heuristic_empirically(["input8_one_process_one_bl_action_unknown_deadlines"], 10000, "Random") # 0.34375
# evaluate_one_heuristic_empirically(["input9_one_process_two_bl_actions_unknown_deadlines"], 10000, "Random")
# evaluate_one_heuristic_empirically(["input10_two_processes_one_bl_action_unknown_deadlines"], 10000, "Random")
# evaluate_one_heuristic_empirically(["input11_three_processes_no_bl_actions_unknown_deadlines"], 10000, "Random")
# evaluate_one_heuristic_empirically(["input12_one_process_one_bl_action_known_deadlines"], 10000, "Random")
# evaluate_one_heuristic_empirically(["input13_one_process_two_bl_actions_known_deadlines"], 10000, "Random")
# evaluate_one_heuristic_empirically(["input14_two_processes_one_bl_action_unknown_deadlines"], 10000, "Random")

# evaluate_one_heuristic_empirically(["input1_one_process_no_bl_actions"], 10000, "MostPromisingPlan") # 0.5
# evaluate_one_heuristic_empirically(["input2_two_processes_no_bl_actions"], 10000, "MostPromisingPlan") # 0.6
# evaluate_one_heuristic_empirically(["input3_one_process_one_bl_action"], 10000, "MostPromisingPlan") # 0.0
# evaluate_one_heuristic_empirically(["input4_one_process_two_bl_actions"], 10000, "MostPromisingPlan") # 0.0
# evaluate_one_heuristic_empirically(["input5_two_processes_one_bl_action"], 10000, "MostPromisingPlan") # 0.8
# evaluate_one_heuristic_empirically(["input6_one_process_no_bl_actions_unknown_deadlines"], 10000, "MostPromisingPlan") # 0.35
# evaluate_one_heuristic_empirically(["input7_two_processes_no_bl_actions_unknown_deadlines"], 10000, "MostPromisingPlan") # 0.51125
# evaluate_one_heuristic_empirically(["input8_one_process_one_bl_action_unknown_deadlines"], 10000, "MostPromisingPlan") # 0.1
# evaluate_one_heuristic_empirically(["input9_one_process_two_bl_actions_unknown_deadlines"], 10000, "MostPromisingPlan") # 0.04
# evaluate_one_heuristic_empirically(["input10_two_processes_one_bl_action_unknown_deadlines"], 10000, "MostPromisingPlan") # 0.8
# evaluate_one_heuristic_empirically(["input11_three_processes_no_bl_actions_unknown_deadlines"], 10000, "MostPromisingPlan") # 0.883
# evaluate_one_heuristic_empirically(["input12_one_process_one_bl_action_known_deadlines"], 10000, "MostPromisingPlan") # 0.0
# evaluate_one_heuristic_empirically(["input13_one_process_two_bl_actions_known_deadlines"], 10000, "MostPromisingPlan") # 0.0
# evaluate_one_heuristic_empirically(["input14_two_processes_one_bl_action_unknown_deadlines"], 10000, "MostPromisingPlan") # 0.85

# evaluate_one_heuristic_empirically(["input1_one_process_no_bl_actions"], 10000, "ExecutiveMostPromisingPlan") # 0.5
# evaluate_one_heuristic_empirically(["input2_two_processes_no_bl_actions"], 10000, "ExecutiveMostPromisingPlan") # 0.6
# evaluate_one_heuristic_empirically(["input3_one_process_one_bl_action"], 10000, "ExecutiveMostPromisingPlan") # 0.5
# evaluate_one_heuristic_empirically(["input4_one_process_two_bl_actions"], 10000, "ExecutiveMostPromisingPlan") # 0.5
# evaluate_one_heuristic_empirically(["input5_two_processes_one_bl_action"], 10000, "ExecutiveMostPromisingPlan") # 0.84
# evaluate_one_heuristic_empirically(["input6_one_process_no_bl_actions_unknown_deadlines"], 10000, "ExecutiveMostPromisingPlan") # 0.35
# evaluate_one_heuristic_empirically(["input7_two_processes_no_bl_actions_unknown_deadlines"], 10000, "ExecutiveMostPromisingPlan") # 0.51125
# evaluate_one_heuristic_empirically(["input8_one_process_one_bl_action_unknown_deadlines"], 10000, "ExecutiveMostPromisingPlan") # 0.5
# evaluate_one_heuristic_empirically(["input9_one_process_two_bl_actions_unknown_deadlines"], 10000, "ExecutiveMostPromisingPlan") # 0.5
# evaluate_one_heuristic_empirically(["input10_two_processes_one_bl_action_unknown_deadlines"], 10000, "ExecutiveMostPromisingPlan") # 0.82
# evaluate_one_heuristic_empirically(["input11_three_processes_no_bl_actions_unknown_deadlines"], 10000, "ExecutiveMostPromisingPlan") # 0.883
# evaluate_one_heuristic_empirically(["input12_one_process_one_bl_action_known_deadlines"], 10000, "ExecutiveMostPromisingPlan") # 0.5
# evaluate_one_heuristic_empirically(["input13_one_process_two_bl_actions_known_deadlines"], 10000, "ExecutiveMostPromisingPlan") # 0.5
# evaluate_one_heuristic_empirically(["input14_two_processes_one_bl_action_unknown_deadlines"], 10000, "ExecutiveMostPromisingPlan") # 0.85

# evaluate_one_heuristic_empirically(["input1_one_process_no_bl_actions"], 10000, "RoundRobin") # 0.5
# evaluate_one_heuristic_empirically(["input2_two_processes_no_bl_actions"], 10000, "RoundRobin") # 0.6
# evaluate_one_heuristic_empirically(["input3_one_process_one_bl_action"], 10000, "RoundRobin") # 0.0
# evaluate_one_heuristic_empirically(["input4_one_process_two_bl_actions"], 10000, "RoundRobin") # 0.0
# evaluate_one_heuristic_empirically(["input5_two_processes_one_bl_action"], 10000, "RoundRobin") # 0.84
# evaluate_one_heuristic_empirically(["input6_one_process_no_bl_actions_unknown_deadlines"], 10000, "RoundRobin") # 0.35
# evaluate_one_heuristic_empirically(["input7_two_processes_no_bl_actions_unknown_deadlines"], 10000, "RoundRobin") # 0.51125
# evaluate_one_heuristic_empirically(["input8_one_process_one_bl_action_unknown_deadlines"], 10000, "RoundRobin") # 0.1
# evaluate_one_heuristic_empirically(["input9_one_process_two_bl_actions_unknown_deadlines"], 10000, "RoundRobin") # 0.04
# evaluate_one_heuristic_empirically(["input10_two_processes_one_bl_action_unknown_deadlines"], 10000, "RoundRobin") # 0.46
# evaluate_one_heuristic_empirically(["input11_three_processes_no_bl_actions_unknown_deadlines"], 10000, "RoundRobin") # 0.688
# evaluate_one_heuristic_empirically(["input12_one_process_one_bl_action_known_deadlines"], 10000, "RoundRobin") # 0.0
# evaluate_one_heuristic_empirically(["input13_one_process_two_bl_actions_known_deadlines"], 10000, "RoundRobin") # 0.0
# evaluate_one_heuristic_empirically(["input14_two_processes_one_bl_action_unknown_deadlines"], 10000, "RoundRobin") # 0.4

# evaluate_one_heuristic_empirically(["input1_one_process_no_bl_actions"], 10000, "ExecutiveRoundRobin") # 0.5
# evaluate_one_heuristic_empirically(["input2_two_processes_no_bl_actions"], 10000, "ExecutiveRoundRobin") # 0.6
# evaluate_one_heuristic_empirically(["input3_one_process_one_bl_action"], 10000, "ExecutiveRoundRobin") # 0.5
# evaluate_one_heuristic_empirically(["input4_one_process_two_bl_actions"], 10000, "ExecutiveRoundRobin") # 0.5
# evaluate_one_heuristic_empirically(["input5_two_processes_one_bl_action"], 10000, "ExecutiveRoundRobin") # 0.84
# evaluate_one_heuristic_empirically(["input6_one_process_no_bl_actions_unknown_deadlines"], 10000, "ExecutiveRoundRobin") # 0.35
# evaluate_one_heuristic_empirically(["input7_two_processes_no_bl_actions_unknown_deadlines"], 10000, "ExecutiveRoundRobin") # 0.51125
# evaluate_one_heuristic_empirically(["input8_one_process_one_bl_action_unknown_deadlines"], 10000, "ExecutiveRoundRobin") # 0.5
# evaluate_one_heuristic_empirically(["input9_one_process_two_bl_actions_unknown_deadlines"], 10000, "ExecutiveRoundRobin") # 0.5
# evaluate_one_heuristic_empirically(["input10_two_processes_one_bl_action_unknown_deadlines"], 10000, "ExecutiveRoundRobin") # 0.35
# evaluate_one_heuristic_empirically(["input11_three_processes_no_bl_actions_unknown_deadlines"], 10000, "ExecutiveRoundRobin") # 0.688
# evaluate_one_heuristic_empirically(["input12_one_process_one_bl_action_known_deadlines"], 10000, "ExecutiveRoundRobin") # 0.5
# evaluate_one_heuristic_empirically(["input13_one_process_two_bl_actions_known_deadlines"], 10000, "ExecutiveRoundRobin") # 0.5
# evaluate_one_heuristic_empirically(["input14_two_processes_one_bl_action_unknown_deadlines"], 10000, "ExecutiveRoundRobin") # 0.35

# evaluate_one_heuristic_empirically(["input1_one_process_no_bl_actions"], 10000, "ExecutiveBasicGreedyScheme", [0]) # 0.5
# evaluate_one_heuristic_empirically(["input2_two_processes_no_bl_actions"], 10000, "ExecutiveBasicGreedyScheme", [0]) # 0.6
# evaluate_one_heuristic_empirically(["input3_one_process_one_bl_action"], 10000, "ExecutiveBasicGreedyScheme", [0]) # 0.5
# evaluate_one_heuristic_empirically(["input4_one_process_two_bl_actions"], 10000, "ExecutiveBasicGreedyScheme", [0]) # 0.5
# evaluate_one_heuristic_empirically(["input5_two_processes_one_bl_action"], 10000, "ExecutiveBasicGreedyScheme", [0]) # 0.84
# evaluate_one_heuristic_empirically(["input6_one_process_no_bl_actions_unknown_deadlines"], 10000, "ExecutiveBasicGreedyScheme", [0]) # 0.35
# evaluate_one_heuristic_empirically(["input7_two_processes_no_bl_actions_unknown_deadlines"], 10000, "ExecutiveBasicGreedyScheme", [0]) # 0.51125
# evaluate_one_heuristic_empirically(["input8_one_process_one_bl_action_unknown_deadlines"], 10000, "ExecutiveBasicGreedyScheme", [0]) # 0.5
# evaluate_one_heuristic_empirically(["input9_one_process_two_bl_actions_unknown_deadlines"], 10000, "ExecutiveBasicGreedyScheme", [0]) # 0.5
# evaluate_one_heuristic_empirically(["input10_two_processes_one_bl_action_unknown_deadlines"], 10000, "ExecutiveBasicGreedyScheme", [0]) # 0.82
# evaluate_one_heuristic_empirically(["input11_three_processes_no_bl_actions_unknown_deadlines"], 10000, "ExecutiveBasicGreedyScheme", [0]) # 0.883
# evaluate_one_heuristic_empirically(["input12_one_process_one_bl_action_known_deadlines"], 10000, "ExecutiveBasicGreedyScheme", [0]) # 0.5
# evaluate_one_heuristic_empirically(["input13_one_process_two_bl_actions_known_deadlines"], 10000, "ExecutiveBasicGreedyScheme", [0]) # 0.5
# evaluate_one_heuristic_empirically(["input14_two_processes_one_bl_action_unknown_deadlines"], 10000, "ExecutiveBasicGreedyScheme", [0]) # 0.85

# evaluate_one_heuristic_empirically(["input1_one_process_no_bl_actions"], 10000, "ExecutiveDDA", [1, 1]) # 0.5
# evaluate_one_heuristic_empirically(["input2_two_processes_no_bl_actions"], 10000, "ExecutiveDDA", [1, 1]) # 0.6
# evaluate_one_heuristic_empirically(["input3_one_process_one_bl_action"], 10000, "ExecutiveDDA", [1, 1]) # 0.5
# evaluate_one_heuristic_empirically(["input4_one_process_two_bl_actions"], 10000, "ExecutiveDDA", [1, 1]) # 0.5
# evaluate_one_heuristic_empirically(["input5_two_processes_one_bl_action"], 10000, "ExecutiveDDA", [1, 1]) # 0.84
# evaluate_one_heuristic_empirically(["input6_one_process_no_bl_actions_unknown_deadlines"], 10000, "ExecutiveDDA", [1, 1]) # 0.35
# evaluate_one_heuristic_empirically(["input7_two_processes_no_bl_actions_unknown_deadlines"], 10000, "ExecutiveDDA", [1, 1]) # 0.51125
# evaluate_one_heuristic_empirically(["input8_one_process_one_bl_action_unknown_deadlines"], 10000, "ExecutiveDDA", [1, 1]) # 0.5
# evaluate_one_heuristic_empirically(["input9_one_process_two_bl_actions_unknown_deadlines"], 10000, "ExecutiveDDA", [1, 1]) # 0.5
# evaluate_one_heuristic_empirically(["input10_two_processes_one_bl_action_unknown_deadlines"], 10000, "ExecutiveDDA", [1, 1]) # 0.82
# evaluate_one_heuristic_empirically(["input11_three_processes_no_bl_actions_unknown_deadlines"], 10000, "ExecutiveDDA", [1, 1]) # 0.883
# evaluate_one_heuristic_empirically(["input12_one_process_one_bl_action_known_deadlines"], 10000, "ExecutiveDDA", [1, 1]) # 0.5
# evaluate_one_heuristic_empirically(["input13_one_process_two_bl_actions_known_deadlines"], 10000, "ExecutiveDDA", [1, 1]) # 0.5
# evaluate_one_heuristic_empirically(["input14_two_processes_one_bl_action_unknown_deadlines"], 10000, "ExecutiveDDA", [1, 1]) # 0.85

# evaluate_one_heuristic_empirically(["input1_one_process_no_bl_actions"], 10000, "DDAOnMaxSize", [1, 1]) # 0.5
# evaluate_one_heuristic_empirically(["input2_two_processes_no_bl_actions"], 10000, "DDAOnMaxSize", [1, 1]) # 0.6
# evaluate_one_heuristic_empirically(["input3_one_process_one_bl_action"], 10000, "DDAOnMaxSize", [1, 1]) # 0.5
# evaluate_one_heuristic_empirically(["input4_one_process_two_bl_actions"], 10000, "DDAOnMaxSize", [1, 1]) # 0.5
# evaluate_one_heuristic_empirically(["input5_two_processes_one_bl_action"], 10000, "DDAOnMaxSize", [1, 1]) # 0.84
# evaluate_one_heuristic_empirically(["input6_one_process_no_bl_actions_unknown_deadlines"], 10000, "DDAOnMaxSize", [1, 1]) # 0.35
# evaluate_one_heuristic_empirically(["input7_two_processes_no_bl_actions_unknown_deadlines"], 10000, "DDAOnMaxSize", [1, 1]) # 0.51125
# evaluate_one_heuristic_empirically(["input8_one_process_one_bl_action_unknown_deadlines"], 10000, "DDAOnMaxSize", [1, 1]) # 0.5
# evaluate_one_heuristic_empirically(["input9_one_process_two_bl_actions_unknown_deadlines"], 10000, "DDAOnMaxSize", [1, 1]) # 0.5
# evaluate_one_heuristic_empirically(["input10_two_processes_one_bl_action_unknown_deadlines"], 10000, "DDAOnMaxSize", [1, 1]) # 0.35
# evaluate_one_heuristic_empirically(["input11_three_processes_no_bl_actions_unknown_deadlines"], 10000, "DDAOnMaxSize", [1, 1]) # 0.883
# evaluate_one_heuristic_empirically(["input12_one_process_one_bl_action_known_deadlines"], 10000, "DDAOnMaxSize", [1, 1]) # 0.5
# evaluate_one_heuristic_empirically(["input13_one_process_two_bl_actions_known_deadlines"], 10000, "DDAOnMaxSize", [1, 1]) # 0.5
# evaluate_one_heuristic_empirically(["input14_two_processes_one_bl_action_unknown_deadlines"], 10000, "DDAOnMaxSize", [1, 1]) # 0.35

# evaluate_one_heuristic_empirically(["input1_one_process_no_bl_actions"], 100, "JustDoSomething", ["SAE2DynamicProgramming", 1]) # 0.5
# evaluate_one_heuristic_empirically(["input2_two_processes_no_bl_actions"], 100, "JustDoSomething", ["SAE2DynamicProgramming", 1]) # 0.5
# evaluate_one_heuristic_empirically(["input3_one_process_one_bl_action"], 100, "JustDoSomething", ["SAE2DynamicProgramming", 1]) # 0.5
# evaluate_one_heuristic_empirically(["input4_one_process_two_bl_actions"], 100, "JustDoSomething", ["SAE2DynamicProgramming", 1]) # 0.5
# evaluate_one_heuristic_empirically(["input5_two_processes_one_bl_action"], 100, "JustDoSomething", ["SAE2DynamicProgramming", 1]) # 0.8
# evaluate_one_heuristic_empirically(["input6_one_process_no_bl_actions_unknown_deadlines"], 100, "JustDoSomething", ["SAE2DynamicProgramming", 1]) # 0.35
# evaluate_one_heuristic_empirically(["input7_two_processes_no_bl_actions_unknown_deadlines"], 100, "JustDoSomething", ["SAE2DynamicProgramming", 1]) # ~0.4.4
# evaluate_one_heuristic_empirically(["input8_one_process_one_bl_action_unknown_deadlines"], 100, "JustDoSomething", ["SAE2DynamicProgramming", 1]) # 0.5
# evaluate_one_heuristic_empirically(["input9_one_process_two_bl_actions_unknown_deadlines"], 100, "JustDoSomething", ["SAE2DynamicProgramming", 1]) # 0.5
# evaluate_one_heuristic_empirically(["input10_two_processes_one_bl_action_unknown_deadlines"], 100, "JustDoSomething", ["SAE2DynamicProgramming", 1]) # 0.8
# evaluate_one_heuristic_empirically(["input11_three_processes_no_bl_actions_unknown_deadlines"], 100, "JustDoSomething", ["SAE2DynamicProgramming", 1]) # ~0.8
# evaluate_one_heuristic_empirically(["input12_one_process_one_bl_action_known_deadlines"], 100, "JustDoSomething", ["SAE2DynamicProgramming", 1]) # 0.5
# evaluate_one_heuristic_empirically(["input13_one_process_two_bl_actions_known_deadlines"], 100, "JustDoSomething", ["SAE2DynamicProgramming", 1]) # 0.0
# evaluate_one_heuristic_empirically(["input14_two_processes_one_bl_action_unknown_deadlines"], 100, "JustDoSomething", ["SAE2DynamicProgramming", 1]) # 0.85

# evaluate_one_heuristic_empirically(["input1_one_process_no_bl_actions"], 1000, "JustDoSomething", ["BasicGreedyScheme", 1, 0]) # 0.5
# evaluate_one_heuristic_empirically(["input2_two_processes_no_bl_actions"], 1000, "JustDoSomething", ["BasicGreedyScheme", 1, 0]) # 0.6
# evaluate_one_heuristic_empirically(["input3_one_process_one_bl_action"], 1000, "JustDoSomething", ["BasicGreedyScheme", 1, 0]) # 0.5
# evaluate_one_heuristic_empirically(["input4_one_process_two_bl_actions"], 1000, "JustDoSomething", ["BasicGreedyScheme", 1, 0]) # 0.5
# evaluate_one_heuristic_empirically(["input5_two_processes_one_bl_action"], 1000, "JustDoSomething", ["BasicGreedyScheme", 1, 0]) # 0.8
# evaluate_one_heuristic_empirically(["input6_one_process_no_bl_actions_unknown_deadlines"], 1000, "JustDoSomething", ["BasicGreedyScheme", 1, 0]) # 0.35
# evaluate_one_heuristic_empirically(["input7_two_processes_no_bl_actions_unknown_deadlines"], 1000, "JustDoSomething", ["BasicGreedyScheme", 1, 0]) # 0.51125
# evaluate_one_heuristic_empirically(["input8_one_process_one_bl_action_unknown_deadlines"], 1000, "JustDoSomething", ["BasicGreedyScheme", 1, 0]) # 0.5
# evaluate_one_heuristic_empirically(["input9_one_process_two_bl_actions_unknown_deadlines"], 1000, "JustDoSomething", ["BasicGreedyScheme", 1, 0]) # 0.5
# evaluate_one_heuristic_empirically(["input10_two_processes_one_bl_action_unknown_deadlines"], 1000, "JustDoSomething", ["BasicGreedyScheme", 1, 0]) # 0.8
# evaluate_one_heuristic_empirically(["input11_three_processes_no_bl_actions_unknown_deadlines"], 1000, "JustDoSomething", ["BasicGreedyScheme", 1, 0]) # 0.883
# evaluate_one_heuristic_empirically(["input12_one_process_one_bl_action_known_deadlines"], 1000, "JustDoSomething", ["BasicGreedyScheme", 1, 0]) # 0.5
# evaluate_one_heuristic_empirically(["input13_one_process_two_bl_actions_known_deadlines"], 1000, "JustDoSomething", ["BasicGreedyScheme", 1, 0]) # 0.5
# evaluate_one_heuristic_empirically(["input14_two_processes_one_bl_action_unknown_deadlines"], 1000, "JustDoSomething", ["BasicGreedyScheme", 1, 0]) # 0.85

# evaluate_one_heuristic_empirically(["input1_one_process_no_bl_actions"], 1000, "JustDoSomething", ["DDA", 1, 1, 1]) # 0.5
# evaluate_one_heuristic_empirically(["input2_two_processes_no_bl_actions"], 1000, "JustDoSomething", ["DDA", 1, 1, 1]) # 0.6
# evaluate_one_heuristic_empirically(["input3_one_process_one_bl_action"], 1000, "JustDoSomething", ["DDA", 1, 1, 1]) # 0.5
# evaluate_one_heuristic_empirically(["input4_one_process_two_bl_actions"], 1000, "JustDoSomething", ["DDA", 1, 1, 1]) # 0.5
# evaluate_one_heuristic_empirically(["input5_two_processes_one_bl_action"], 1000, "JustDoSomething", ["DDA", 1, 1, 1]) # 0.8
# evaluate_one_heuristic_empirically(["input6_one_process_no_bl_actions_unknown_deadlines"], 1000, "JustDoSomething", ["DDA", 1, 1, 1]) # 0.35
# evaluate_one_heuristic_empirically(["input7_two_processes_no_bl_actions_unknown_deadlines"], 1000, "JustDoSomething", ["DDA", 1, 1, 1]) # 0.51125
# evaluate_one_heuristic_empirically(["input8_one_process_one_bl_action_unknown_deadlines"], 1000, "JustDoSomething", ["DDA", 1, 1, 1]) # 0.5
# evaluate_one_heuristic_empirically(["input9_one_process_two_bl_actions_unknown_deadlines"], 1000, "JustDoSomething", ["DDA", 1, 1, 1]) # 0.5
# evaluate_one_heuristic_empirically(["input10_two_processes_one_bl_action_unknown_deadlines"], 1000, "JustDoSomething", ["DDA", 1, 1, 1]) # 0.8
# evaluate_one_heuristic_empirically(["input11_three_processes_no_bl_actions_unknown_deadlines"], 1000, "JustDoSomething", ["DDA", 1, 1, 1]) # 0.883
# evaluate_one_heuristic_empirically(["input12_one_process_one_bl_action_known_deadlines"], 1000, "JustDoSomething", ["DDA", 1, 1, 1]) # 0.5
# evaluate_one_heuristic_empirically(["input13_one_process_two_bl_actions_known_deadlines"], 1000, "JustDoSomething", ["DDA", 1, 1, 1]) # 0.5
# evaluate_one_heuristic_empirically(["input14_two_processes_one_bl_action_unknown_deadlines"], 1000, "JustDoSomething", ["DDA", 1, 1, 1]) # 0.85

# evaluate_one_heuristic_empirically(["input1_one_process_no_bl_actions"], 1000, "Laziness", ["DDA", 1, 1, 1]) # 0.5
# evaluate_one_heuristic_empirically(["input2_two_processes_no_bl_actions"], 1000, "Laziness", ["DDA", 1, 1, 1]) # 0.6
# evaluate_one_heuristic_empirically(["input3_one_process_one_bl_action"], 1000, "Laziness", ["DDA", 1, 1, 1]) # 0.5
# evaluate_one_heuristic_empirically(["input4_one_process_two_bl_actions"], 1000, "Laziness", ["DDA", 1, 1, 1]) # 0.5
# evaluate_one_heuristic_empirically(["input5_two_processes_one_bl_action"], 1000, "Laziness", ["DDA", 1, 1, 1]) # 0.8
# evaluate_one_heuristic_empirically(["input6_one_process_no_bl_actions_unknown_deadlines"], 1000, "Laziness", ["DDA", 1, 1, 1]) # 0.35
# evaluate_one_heuristic_empirically(["input7_two_processes_no_bl_actions_unknown_deadlines"], 1000, "Laziness", ["DDA", 1, 1, 1]) # 0.51125
# evaluate_one_heuristic_empirically(["input8_one_process_one_bl_action_unknown_deadlines"], 1000, "Laziness", ["DDA", 1, 1, 1]) # 0.5
# evaluate_one_heuristic_empirically(["input9_one_process_two_bl_actions_unknown_deadlines"], 1000, "Laziness", ["DDA", 1, 1, 1]) # 0.5
# evaluate_one_heuristic_empirically(["input10_two_processes_one_bl_action_unknown_deadlines"], 1000, "Laziness", ["DDA", 1, 1, 1]) # 0.8
# evaluate_one_heuristic_empirically(["input11_three_processes_no_bl_actions_unknown_deadlines"], 1000, "Laziness", ["DDA", 1, 1, 1]) # 0.883
# evaluate_one_heuristic_empirically(["input12_one_process_one_bl_action_known_deadlines"], 1000, "Laziness", ["DDA", 1, 1, 1]) # 0.5
# evaluate_one_heuristic_empirically(["input13_one_process_two_bl_actions_known_deadlines"], 1000, "Laziness", ["DDA", 1, 1, 1]) # 0.5
# evaluate_one_heuristic_empirically(["input14_two_processes_one_bl_action_unknown_deadlines"], 1000, "Laziness", ["DDA", 1, 1, 1]) # 0.85

# evaluate_one_heuristic_empirically(["input1_one_process_no_bl_actions"], 100, "Laziness", ["SAE2DynamicProgramming"]) # 0.5
# evaluate_one_heuristic_empirically(["input2_two_processes_no_bl_actions"], 100, "Laziness", ["SAE2DynamicProgramming"]) # 0.6
# evaluate_one_heuristic_empirically(["input3_one_process_one_bl_action"], 100, "Laziness", ["SAE2DynamicProgramming"]) # 0.5
# evaluate_one_heuristic_empirically(["input4_one_process_two_bl_actions"], 100, "Laziness", ["SAE2DynamicProgramming"]) # 0.5
# evaluate_one_heuristic_empirically(["input5_two_processes_one_bl_action"], 100, "Laziness", ["SAE2DynamicProgramming"]) # 0.8
# evaluate_one_heuristic_empirically(["input6_one_process_no_bl_actions_unknown_deadlines"], 100, "Laziness", ["SAE2DynamicProgramming"]) # 0.35
# evaluate_one_heuristic_empirically(["input7_two_processes_no_bl_actions_unknown_deadlines"], 100, "Laziness", ["SAE2DynamicProgramming"]) # ~0.4.4
# evaluate_one_heuristic_empirically(["input8_one_process_one_bl_action_unknown_deadlines"], 100, "Laziness", ["SAE2DynamicProgramming"]) # 0.5
# evaluate_one_heuristic_empirically(["input9_one_process_two_bl_actions_unknown_deadlines"], 100, "Laziness", ["SAE2DynamicProgramming"]) # 0.5
# evaluate_one_heuristic_empirically(["input10_two_processes_one_bl_action_unknown_deadlines"], 100, "Laziness", ["SAE2DynamicProgramming"]) # 0.8
# evaluate_one_heuristic_empirically(["input11_three_processes_no_bl_actions_unknown_deadlines"], 100, "Laziness", ["SAE2DynamicProgramming"]) # ~0.8
# evaluate_one_heuristic_empirically(["input12_one_process_one_bl_action_known_deadlines"], 100, "Laziness", ["SAE2DynamicProgramming"]) # 0.5
# evaluate_one_heuristic_empirically(["input13_one_process_two_bl_actions_known_deadlines"], 100, "Laziness", ["SAE2DynamicProgramming"]) # 0.5
# evaluate_one_heuristic_empirically(["input14_two_processes_one_bl_action_unknown_deadlines"], 100, "Laziness", ["SAE2DynamicProgramming"]) # 0.85

# evaluate_one_heuristic_empirically(["input1_one_process_no_bl_actions"], 100, "MCTS", [10])
# evaluate_one_heuristic_empirically(["input2_two_processes_no_bl_actions"], 100, "MCTS", [10])
# evaluate_one_heuristic_empirically(["input3_one_process_one_bl_action"], 100, "MCTS", [10])
# evaluate_one_heuristic_empirically(["input4_one_process_two_bl_actions"], 100, "MCTS", [10])
# evaluate_one_heuristic_empirically(["input5_two_processes_one_bl_action"], 100, "MCTS", [10])
# evaluate_one_heuristic_empirically(["input6_one_process_no_bl_actions_unknown_deadlines"], 100, "MCTS", [10])
# evaluate_one_heuristic_empirically(["input7_two_processes_no_bl_actions_unknown_deadlines"], 100, "MCTS", [10])
# evaluate_one_heuristic_empirically(["input8_one_process_one_bl_action_unknown_deadlines"], 100, "MCTS", [10])
# evaluate_one_heuristic_empirically(["input9_one_process_two_bl_actions_unknown_deadlines"], 100, "MCTS", [10])
# evaluate_one_heuristic_empirically(["input10_two_processes_one_bl_action_unknown_deadlines"], 100, "MCTS", [10])
# evaluate_one_heuristic_empirically(["input11_three_processes_no_bl_actions_unknown_deadlines"], 100, "MCTS", [10])
# evaluate_one_heuristic_empirically(["input12_one_process_one_bl_action_known_deadlines"], 100, "MCTS", [10])
# evaluate_one_heuristic_empirically(["input13_one_process_two_bl_actions_known_deadlines"], 100, "MCTS", [10])
# evaluate_one_heuristic_empirically(["input14_two_processes_one_bl_action_unknown_deadlines"], 100, "MCTS", [10])

# evaluate_one_heuristic_empirically(["input1_one_process_no_bl_actions"], 1000, "KBounded", [3, "DDA", [1, 1, 1]])
# evaluate_one_heuristic_empirically(["input2_two_processes_no_bl_actions"], 1000, "KBounded", [3, "DDA", [1, 1, 1]])
# evaluate_one_heuristic_empirically(["input3_one_process_one_bl_action"], 1000, "KBounded", [3, "DDA", [1, 1, 1]])
# evaluate_one_heuristic_empirically(["input4_one_process_two_bl_actions"], 1000, "KBounded", [3, "DDA", [1, 1, 1]])
# evaluate_one_heuristic_empirically(["input5_two_processes_one_bl_action"], 1000, "KBounded", [3, "DDA", [1, 1, 1]])
# evaluate_one_heuristic_empirically(["input6_one_process_no_bl_actions_unknown_deadlines"], 1000, "KBounded", [3, "DDA", [1, 1, 1]])
# evaluate_one_heuristic_empirically(["input7_two_processes_no_bl_actions_unknown_deadlines"], 1000, "KBounded", [3, "DDA", [1, 1, 1]])
# evaluate_one_heuristic_empirically(["input8_one_process_one_bl_action_unknown_deadlines"], 1000, "KBounded", [3, "DDA", [1, 1, 1]])
# evaluate_one_heuristic_empirically(["input9_one_process_two_bl_actions_unknown_deadlines"], 1000, "KBounded", [3, "DDA", [1, 1, 1]])
# evaluate_one_heuristic_empirically(["input10_two_processes_one_bl_action_unknown_deadlines"], 1000, "KBounded", [3, "DDA", [1, 1, 1]])
# evaluate_one_heuristic_empirically(["input11_three_processes_no_bl_actions_unknown_deadlines"], 1000, "KBounded", [3, "DDA", [1, 1, 1]])
# evaluate_one_heuristic_empirically(["input12_one_process_one_bl_action_known_deadlines"], 1000, "KBounded", [3, "DDA", [1, 1, 1]])
# evaluate_one_heuristic_empirically(["input13_one_process_two_bl_actions_known_deadlines"], 1000, "KBounded", [3, "DDA", [1, 1, 1]])
# evaluate_one_heuristic_empirically(["input14_two_processes_one_bl_action_unknown_deadlines"], 1000, "KBounded", [3, "DDA", [1, 1, 1]])

# evaluate_one_heuristic_empirically(["input1_one_process_no_bl_actions"], 1000, "KBounded", [3, "SAE2DynamicProgramming"])
# evaluate_one_heuristic_empirically(["input2_two_processes_no_bl_actions"], 1000, "KBounded", [3, "SAE2DynamicProgramming"])
# evaluate_one_heuristic_empirically(["input3_one_process_one_bl_action"], 1000, "KBounded", [3, "SAE2DynamicProgramming"])
# evaluate_one_heuristic_empirically(["input4_one_process_two_bl_actions"], 1000, "KBounded", [3, "SAE2DynamicProgramming"])
# evaluate_one_heuristic_empirically(["input5_two_processes_one_bl_action"], 1000, "KBounded", [3, "SAE2DynamicProgramming"])
# evaluate_one_heuristic_empirically(["input6_one_process_no_bl_actions_unknown_deadlines"], 1000, "KBounded", [3, "SAE2DynamicProgramming"])
# evaluate_one_heuristic_empirically(["input7_two_processes_no_bl_actions_unknown_deadlines"], 1000, "KBounded", [3, "SAE2DynamicProgramming"])
# evaluate_one_heuristic_empirically(["input8_one_process_one_bl_action_unknown_deadlines"], 1000, "KBounded", [3, "SAE2DynamicProgramming"])
# evaluate_one_heuristic_empirically(["input9_one_process_two_bl_actions_unknown_deadlines"], 1000, "KBounded", [3, "SAE2DynamicProgramming"])
# evaluate_one_heuristic_empirically(["input10_two_processes_one_bl_action_unknown_deadlines"], 1000, "KBounded", [3, "SAE2DynamicProgramming"])
# evaluate_one_heuristic_empirically(["input11_three_processes_no_bl_actions_unknown_deadlines"], 1000, "KBounded", [3, "SAE2DynamicProgramming"])
# evaluate_one_heuristic_empirically(["input12_one_process_one_bl_action_known_deadlines"], 1000, "KBounded", [3, "SAE2DynamicProgramming"])
# evaluate_one_heuristic_empirically(["input13_one_process_two_bl_actions_known_deadlines"], 1000, "KBounded", [3, "SAE2DynamicProgramming"])
# evaluate_one_heuristic_empirically(["input14_two_processes_one_bl_action_unknown_deadlines"], 1000, "KBounded", [3, "SAE2DynamicProgramming"])





# # folder = "ipae_input_bank/5_5_kd_20"
# # folder = "ipae_input_bank/5_10_kd_20"
# # folder = "ipae_input_bank/5_20_kd_20"
# # folder = "ipae_input_bank/10_5_kd_10"
# # folder = "ipae_input_bank/3_10_kd_10"
# # folder = "ipae_input_bank/3_20_kd_10"
# # folder = "ipae_input_bank/5_30_kd_10"
# # folder = "ipae_input_bank/5_100_kd_10"
# # folder = "ipae_input_bank/10_200_kd_10"
# # folder = "ipae_input_bank/test_generator"
# inputs = os.listdir(folder)
# # inputs = ['input0']
# runs = 5
# runtime_averages, success_ratios = \
#     evaluate_all_heuristics_empirically(folder, inputs, runs, 0, 1, 1, 1, 1, "DDA", 1, 1, 1, "DDA", 1, 1, 1)
# # for i in range(len(inputs)):
# #     print(runtime_averages[i])
# # for i in range(len(inputs)):
# #     print(success_ratios[i])
# num_of_inputs = len(inputs)
# all_heuristics = ['mpp', 'empp', 'rr', 'err', 'bgs', 'ebgs', 'dda', 'edda', 'dda_ml', 'laz']
# # all_heuristics = ['mpp', 'empp', 'rr', 'err', 'bgs', 'ebgs', 'dda', 'edda', 'dda_ml', 'jds', 'laz']
# for i in range(len(all_heuristics)):
#     print(all_heuristics[i] + ": " + str(sum([ratios[i] for ratios in success_ratios]) / num_of_inputs) + ", " +
#           str(sum([ratios[i] for ratios in runtime_averages])))


# processes, blas, real_tts, real_dls = build_input_from_text_file("ipae_input_bank/ex1") # [2],[0,1],[0]
# processes, blas, real_tts, real_dls = build_input_from_text_file("ipae_input_bank/ex2") # [0,2],[0,1],[0]
# processes, blas, real_tts, real_dls = build_input_from_text_file("ipae_input_bank/ex3") # [2],[0,1]
# processes, blas, real_tts, real_dls = build_input_from_text_file("ipae_input_bank/ex4") # [0,2],[1]
# from ipae_heuristics import compute_forward_compatibility_groups
# groups = compute_forward_compatibility_groups([0,[0,0,0],0,0], processes, 1)
# for g in groups:
#     print([p.serial_number for p in g])



def shortened_name(heuristic_full_name_and_params):
    heuristic_full_name = heuristic_full_name_and_params[0]
    if heuristic_full_name == "MDP":
        return "mdp"
    if heuristic_full_name == "Schedule":
        return "scdl"
    if heuristic_full_name == "SAE2DynamicProgramming":
        return "dp"
    if heuristic_full_name == "Random":
        return "rand"
    if heuristic_full_name == "MostPromisingPlan":
        return "mpp"
    if heuristic_full_name == "ExecutiveMostPromisingPlan":
        return "empp"
    if heuristic_full_name == "RoundRobin":
        return "rr"
    if heuristic_full_name == "ExecutiveRoundRobin":
        return "err"
    if heuristic_full_name == "BasicGreedyScheme":
        return "bgs"
    if heuristic_full_name == "ExecutiveBasicGreedyScheme":
        return "ebgs"
    if heuristic_full_name == "DDA":
        return "dda"
    if heuristic_full_name == "ExecutiveDDA":
        return "edda"
    if heuristic_full_name == "DDAOnMaxSize":
        return "dda_ms"
    if heuristic_full_name == "JustDoSomething":
        return "jds" + "+" + shortened_name([heuristic_full_name_and_params[1][0]])
    if heuristic_full_name == "Laziness":
        return "laz" + "+" + shortened_name([heuristic_full_name_and_params[1]])
    if heuristic_full_name == "MCTS":
        return "mcts" + "+" + str(heuristic_full_name_and_params[1])
    if heuristic_full_name == "KBounded":
        return str(heuristic_full_name_and_params[1]) + "-bnd" + "+" + shortened_name([heuristic_full_name_and_params[2]])

def test_heuristics_on_inputs(folder, inputs, runs, he_names_and_params):
    runtime_averages, success_ratios = evaluate_all_heuristics_empirically(folder, inputs, runs, he_names_and_params)
    num_of_inputs = len(inputs)
    all_heuristics = [shortened_name(hnap) for hnap in he_names_and_params]
    for i in range(len(all_heuristics)):
        print(all_heuristics[i] + ": " + str(sum([ratios[i] for ratios in success_ratios]) / num_of_inputs) + ", " +
            str(sum([runtimes[i] / num_of_inputs for runtimes in runtime_averages]))[:4])


# folder = "ipae_input_bank/5_5_kd_20"
# folder = "ipae_input_bank/5_10_kd_20"
# folder = "ipae_input_bank/5_20_kd_20"
# folder = "ipae_input_bank/10_5_kd_10"
# folder = "ipae_input_bank/3_10_kd_10"
# folder = "ipae_input_bank/3_20_kd_10"
# folder = "ipae_input_bank/5_30_kd_10"
# folder = "ipae_input_bank/5_100_kd_10"
# folder = "ipae_input_bank/10_200_kd_10"
# folder = "ipae_input_bank/test_generator"
folder = "ipae_input_bank/inputs_for_debugging"
# folder = "15-puzzle/15_puzzle_inputs/100_10_5"
# folder = "15-puzzle/15_puzzle_inputs/100_10_10"
# folder = "15-puzzle/15_puzzle_inputs/100_10_15"
# folder = "15-puzzle/15_puzzle_inputs/100_10_20"
# folder = "15-puzzle/15_puzzle_inputs/100_20_5"
# folder = "15-puzzle/15_puzzle_inputs/100_20_10"
# folder = "15-puzzle/15_puzzle_inputs/100_20_15"
# folder = "15-puzzle/15_puzzle_inputs/100_20_20"
# folder = "15-puzzle/15_puzzle_inputs/100_2_1"
# folder = "15-puzzle/15_puzzle_inputs/100_2_2"
# folder = "15-puzzle/15_puzzle_inputs/100_2_3"
# folder = "15-puzzle/15_puzzle_inputs/100_2_5"
# folder = "15-puzzle/15_puzzle_inputs/100_2_10"
# folder = "15-puzzle/15_puzzle_inputs/100_5_1"
# folder = "15-puzzle/15_puzzle_inputs/100_5_2"
# folder = "15-puzzle/15_puzzle_inputs/100_5_3"
# folder = "15-puzzle/15_puzzle_inputs/100_5_5"
# folder = "15-puzzle/15_puzzle_inputs/100_5_10"
# folder = "15-puzzle/15_puzzle_inputs/100_10_1"
# folder = "15-puzzle/15_puzzle_inputs/100_10_2"
# folder = "15-puzzle/15_puzzle_inputs/100_10_3"
# folder = "15-puzzle/15_puzzle_inputs/100_10_5"
# folder = "15-puzzle/15_puzzle_inputs/100_10_10"
# folder = "15-puzzle/15_puzzle_inputs/100_20_1"
# folder = "15-puzzle/15_puzzle_inputs/100_20_2"
# folder = "15-puzzle/15_puzzle_inputs/100_20_3"
# folder = "15-puzzle/15_puzzle_inputs/100_20_5"
# folder = "15-puzzle/15_puzzle_inputs/100_20_10"
# folder = "15-puzzle/15_puzzle_inputs/100_50_1"
# folder = "15-puzzle/15_puzzle_inputs/100_50_2"
# folder = "15-puzzle/15_puzzle_inputs/100_50_3"
# folder = "15-puzzle/15_puzzle_inputs/100_50_5"
# folder = "15-puzzle/15_puzzle_inputs/100_50_10"
# folder = "15-puzzle/15_puzzle_inputs/100_100_1"
# folder = "15-puzzle/15_puzzle_inputs/100_100_2"
# folder = "15-puzzle/15_puzzle_inputs/100_100_3"
# folder = "15-puzzle/15_puzzle_inputs/100_100_5"
# folder = "15-puzzle/15_puzzle_inputs/100_100_10"
# folder = "15-puzzle/15_puzzle_inputs/helper_folder"
# folder = "15-puzzle/15_puzzle_inputs/helper_folder2"
inputs = os.listdir(folder)
# inputs = ['input21']
# inputs = inputs[:5]
# inputs = ['bgs_vs_dda']
runs = 1
he_names_and_params = [
                       # # ["MDP", None, None],
                       # ["SAE2DynamicProgramming", None],
                       # ["Random", None],
                       # ["MostPromisingPlan", None],
                       # ["ExecutiveMostPromisingPlan", None],
                       # ["RoundRobin", None],
                       # ["ExecutiveRoundRobin", None],
                       ["BasicGreedyScheme", 0],
                       # ["ExecutiveBasicGreedyScheme", 0],
                       ["DDA", 1, 1],
                       # ["ExecutiveDDA", 1, 1],
                       # ["DDAOnMaxSize", 1, 1],
                       # ["JustDoSomething", ["SAE2DynamicProgramming", 1, None]],
                       # ["JustDoSomething", ["BasicGreedyScheme", 1, 0]],
                       # ["JustDoSomething", ["DDA", 1, 1, 1]],
                       # ["Laziness", "SAE2DynamicProgramming", None],
                       # ["Laziness", "BasicGreedyScheme", 0],
                       # ["Laziness", "DDA", 1, 1],
                       # ["MCTS", 10],
                       # ["MCTS", 50],
                       # ["MCTS", 100],
                       # ["MCTS", 200],
                       # ["MCTS", 500],
                       # ["MCTS", 1000],
                       # ["KBounded", 2, "SAE2DynamicProgramming", [None]],
                       # ["KBounded", 2, "BasicGreedyScheme", [0]],
                       # ["KBounded", 2, "DDA", [1, 1]],
                       # ["KBounded", 3, "SAE2DynamicProgramming", [None]],
                       # ["KBounded", 3, "BasicGreedyScheme", [0]],
                       # ["KBounded", 3, "DDA", [1, 1]],
                      ]
test_heuristics_on_inputs(folder, inputs, runs, he_names_and_params)



def print_optimal_probabilities(folder, inputs):
    opt_probs, opt_comp_times = [[] for _ in range(len(inputs))], [[] for _ in range(len(inputs))]
    for i, input in enumerate(inputs):
        opt_probs[i], opt_comp_times[i] = compute_optimal_probability_known_deadlines_file_input(folder, input)
        print("opt: " + str(opt_probs[i]) + ", opt_comp_time: " + str(opt_comp_times[i]))
        print()
    print("average probability:", sum(opt_probs) / len(opt_probs))
    print("average computation time:", sum(opt_comp_times) / len(opt_comp_times))


# folder = "ipae_input_bank/5_10_kd_20"
folder = "ipae_input_bank/test_generator"
inputs = os.listdir(folder)
# inputs = ['input9']
# print_optimal_probabilities(folder, inputs)



# TODO:

# - check why the function compute_optimal_probability_known_deadlines returns different probabilities for the same inputs on
#   different runs

# - in the new input generator, fix the issue that causes 

# - organize the files

# - erase the commented out commands in SAE2DynamicProgrammingHeuristic someday

# - the next exponential algorithm is based on way to set the times of blsa when given a series of blas and known allocations such
#   that there is a way to order them without making any of the computation actions tardy.
#   it might be a good idea to right it down in the PDF
# - exponential algorithm (maybe step 3 should be refined):
#   1. pass over the processes in some order and allocate them computation time like in a DP
#   2. for any such allocation, try to find best timing for p.H for all process p and compute the probability of success
#   3. return the best schedule

# - given an input, consider just a subset of processes, which are the best in some sense
#   - one option is most promising group:
#   1. choose some number k of blas which is not too low (depends on statistics over branching factor).
#   2. find the groups of processes that compatible in the first k blas.
#   3. for any such group use eager execution times for the k common actions and be optimistic for the rest blas.
#   4. optional: choose the best group according to step 3 and repeat the process with k' <= k blas, but lazy after the first k.

# - build a sandwich evalutaion heuristic for states



# - for the new input generator:
#       1. bugs: a. the distributions sometimes get a positive probability to time 0
#                b. the subtracting of the 0.5 in the function sample_real_times seems to be in the wrong coordinate, and maybe
#                   the comment line is the correct way to do that
#       2. when building input by file, use the real deadlines and real termination times that appear in the file
#       3. it doesn't create small numbers because I put just 15 values for p.d and 10 for p.m, but the 10 is for 0.5
#       4. there might be a better way to restrict the probability except for normalizing to 0.5 total and add 0.5 to last option
#       5. the inputs don't build a beautiful tree, and this may help EMPP, or any computation-aimed heuristic, to be much
#          closer to JDS

# - try to extract input files out of JSON files (currently S(AE)^2 inputs, since the partial plans don't appear in the JSON files)

# - create a temporal input bank
# - (?) write methods that create processes using a distribution
# - measure the running time of the MDP in different cases

# - (?) write a heuristic that optimizes the base-level actions and their execution times in a similar way to that of N-queens



# - compress distributions (shorten computations):
#       take the k best e_values* of a process and rewrite it's m distribution accordingly. Afterwards d can be compressed too by
#       taking the time before between and after new times of m.
#   * by e_value I mean the ratio between time and probability.
#   example for m: if d = [1.0 : 4] and m = [0.1 : 1, 0.3 : 2, 0.5 : 3, 0.1 : 5] then the e_values are [0.1, 0.2, 0.3, 0]. If we
#       want to take only the 2 higher e_values, we need to ignore the first. If so, the probability of 1 should be added to the
#       probability of 2 and we get [0.4 : 2, 0.5 : 3, 0.1 : 5] and e_values of [0.2, 0.3, 0].
#   example for d: if the compressed m is [... 0.3 : 5, 0.2 : 8 ...], then if d is [... 0.2 : 6, 0.1 : 7 ...], d can be compressed
#       to [... 0.3 : 7 ...] (or 0.3 : 6, nevermind).

# - to prevent repititions of reductions to S(AE)^2, I can use a list of (BLA, t) pairs that keeps the reductions that have already
#   been made



# - (maybe irrelevant: the sandwich heuristic seems to be suitable for measuring state only, but there may be some way to use it for
#   algorithm except for JDS)
# - build SandwichHeuristic that as a beginning computes for all processes:
#       1. a pessimistic (admissible) heuristic.
#       2. an optimistic (inadmissible) heuristic - use the reductive function above + DDA (DDA replaces an optimal heuristic).
#       3. the probabilities of the heuristics (and the difference between the probabilities).
#   then, given the probabilities (and the differences) decide somehow which schedule is the best and keep it, and return the actions
#   one by one as in Schedule (i.e. the Schedule heuristic).
#   (use functions for steps 1-3 to be able to make changes in copies easily)

# - (?) debug compute_forward_compatibility_groups
#       use compute_forward_compatibility_groups for heuristics that used compute_compatibility_groups

# - write a heuristic that evaluates the quailty of a group as follows: it computes the Q-values of all the processes now, then it
#   uses EDDA for the next t_u time units, and then it computes the Q-values for after t_u time units and uses again EDDA and so on.

# - think about evaluations for groups of processes
#   Note: in the unknown deadlines case a lazy execution is not good enough (see differences between ExecutiveDDA and DDAOnMaxSize).
#         One way to deal with it may be giving weights to the process according to the first deadline: if the minimal relevant
#         deadline is d_min with probability p, the weight of this process will be p, and then the group with maximal weight is the
#         best (take care on the fact that the distributions should be normalized according to the current time for p to be the
#         right weight). The weights of the processes in the last group (no execution of a base-level action) should be 1.

# - (?) evaluate heuristic in the case of unknown deadlines