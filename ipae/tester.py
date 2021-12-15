import os

from ipae_heuristics import evaluate_all_heuristics_empirically

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
# folder = "15-puzzle/15_puzzle_inputs/100_10_5"
# folder = "15-puzzle/15_puzzle_inputs/100_10_10"
# folder = "15-puzzle/15_puzzle_inputs/100_10_15"
# folder = "15-puzzle/15_puzzle_inputs/100_10_20"
# folder = "15-puzzle/15_puzzle_inputs/100_20_5"
# folder = "15-puzzle/15_puzzle_inputs/100_20_10"
# folder = "15-puzzle/15_puzzle_inputs/100_20_15"
# folder = "15-puzzle/15_puzzle_inputs/100_20_20"
folder = "15-puzzle/15_puzzle_inputs/100_2_1"
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
# inputs = ['input1']
# inputs = inputs[90:]
runs = 1
he_names_and_params = [
                    # # ["MDP", None, None],
                    #    ["SAE2DynamicProgramming", None],
                    #    ["Random", None],
                    #    ["MostPromisingPlan", None],
                    #    ["ExecutiveMostPromisingPlan", None],
                    #    ["RoundRobin", None],
                    #    ["ExecutiveRoundRobin", None],
                    #    ["BasicGreedyScheme", 0],
                    #    ["ExecutiveBasicGreedyScheme", 0],
                    #    ["DDA", 1, 1],
                    #    ["ExecutiveDDA", 1, 1],
                    #    ["DDAOnMaxSize", 1, 1],
                    # #    ["JustDoSomething", ["SAE2DynamicProgramming", 1, None]],
                    #    ["JustDoSomething", ["BasicGreedyScheme", 1, 0]],
                    #    ["JustDoSomething", ["DDA", 1, 1, 1]],
                    #    ["Laziness", "SAE2DynamicProgramming", None],
                    #    ["Laziness", "BasicGreedyScheme", 0],
                    #    ["Laziness", "DDA", 1, 1],
                    #    ["MCTS", 10],
                    #    ["MCTS", 50],
                    #    ["MCTS", 100],
                    #    ["MCTS", 200],
                    #    ["MCTS", 500],
                    #    ["MCTS", 1000],
                    #    ["KBounded", 2, "SAE2DynamicProgramming", [None]],
                    #    ["KBounded", 2, "BasicGreedyScheme", [0]],
                    #    ["KBounded", 2, "DDA", [1, 1]],
                    #    ["KBounded", 3, "SAE2DynamicProgramming", [None]],
                    #    ["KBounded", 3, "BasicGreedyScheme", [0]],
                    #    ["KBounded", 3, "DDA", [1, 1]],
                      ]
test_heuristics_on_inputs(folder, inputs, runs, he_names_and_params)

# def test(folders, he_names_and_params):
#     for i in range(len(folders)):
#         folder = folders[i]
#         inputs = os.listdir(folder)
#         hnap = he_names_and_params[i]
#         test_heuristics_on_inputs(folder, inputs, runs, hnap)
#
#
# folders = []
# hnaps = []
# folders.append("15-puzzle/15_puzzle_inputs/100_2_1")
# tmp_hnaps = []
# tmp_hnaps.append(["MostPromisingPlan", None])
# tmp_hnaps.append(["ExecutiveMostPromisingPlan", None])
# hnaps.append(tmp_hnaps)
# folders.append("15-puzzle/15_puzzle_inputs/100_2_2")
# tmp_hnaps = []
# tmp_hnaps.append(["MostPromisingPlan", None])
# tmp_hnaps.append(["ExecutiveMostPromisingPlan", None])
# hnaps.append(tmp_hnaps)
# test(folders, hnaps)