import os
import re
from ipae_heuristics import *


def build_heuristic(heuristic_name, processes, base_level_actions, additional_params=None):
    if heuristic_name == "MDP":
        if additional_params[0] is None:
            additional_params[0] = 0.9
        if additional_params[1] is None:
            additional_params[0] = True
        heuristic = MDPHeuristic(processes, base_level_actions, additional_params[0], additional_params[1])
    if heuristic_name == "Schedule":
        for j, a in enumerate(additional_params[0]):
            if isinstance(a, str):
                additional_params[0][j] = base_level_actions[int(a)]
        heuristic = Schedule(additional_params[0])
    elif heuristic_name == "SAE2DynamicProgramming":
        heuristic = SAE2DynamicProgramming(processes)
    elif heuristic_name == "Random":
        heuristic = RandomHeuristic(processes, base_level_actions)
    elif heuristic_name == "MostPromisingPlan":
        heuristic = MostPromisingPlanHeuristic(processes)
    elif heuristic_name == "ExecutiveMostPromisingPlan":
        heuristic = ExecutiveMostPromisingPlanHeuristic(processes)
    elif heuristic_name == "RoundRobin":
        heuristic = RoundRobinHeuristic(processes)
    elif heuristic_name == "ExecutiveRoundRobin":
        heuristic = ExecutiveRoundRobinHeuristic(processes)
    elif heuristic_name == "BasicGreedyScheme":
        heuristic = BasicGreedySchemeHeuristic(processes, additional_params[0])
    elif heuristic_name == "ExecutiveBasicGreedyScheme":
        heuristic = ExecutiveBasicGreedySchemeHeuristic(processes, additional_params[0])
    elif heuristic_name == "DDA":
        heuristic = DDAHeuristic(processes, additional_params[0], additional_params[1])
        # heuristic = DDAHeuristic3(processes, additional_params[0], additional_params[1])
    elif heuristic_name == "ExecutiveDDA":
        heuristic = ExecutiveDDAHeuristic(processes, additional_params[0], additional_params[1])
    elif heuristic_name == "DDAOnMaxSize":
        heuristic = DDAOnMaxSizeHeuristic(processes, base_level_actions, additional_params[0], additional_params[1])
    elif heuristic_name == "JustDoSomething":
        heuristic = JustDoSomethingHeuristic(additional_params[0][0], processes, base_level_actions, additional_params[0][1],
                                             additional_params[0][2:])
    elif heuristic_name == "Laziness":
        heuristic = LazinessHeuristic(processes, base_level_actions, additional_params[0], additional_params[1:])
    elif heuristic_name == "MCTS":
        if len(additional_params) == 1: # the internal heuristic doesn't necessarily require input for the constant C
            additional_params.append(math.sqrt(2))
        heuristic = MCTSHeuristic(processes, additional_params[0], additional_params[1])
    elif heuristic_name == "KBounded":
        if len(additional_params) == 2: # the internal heuristic doesn't necessarily require additional parameters
            additional_params.append(None)
        heuristic = KBoundedHeuristic(processes, base_level_actions, additional_params[0], additional_params[1],
                                        additional_params[2])
    return heuristic

def build_dist_from_line(line):
    '''parse a string and create a PMF'''
    line_numbers = [float(f) for f in re.findall(r'\d+(?:\.\d+)?', line)]
    dist_list = []
    for i in range(0, len(line_numbers), 2):
        dist_list.append((line_numbers[i], int(line_numbers[i+1])))
    return Distribution(dist_list)

# def build_lists_of_termination_times_and_deadlines(file_name):
#     if len(file_name) < 5 or file_name[-4:] !=  ".txt":
#         file_name += ".txt"
#     file = open(file_name)
#     try:
#         lines = file.readlines()
#         real_termination_times = [int(tt) for tt in re.findall(r'\d+', lines[2])]
#         real_deadlines = [int(dl) for dl in re.findall(r'\d+', lines[3])]
#     finally:
#         file.close()
#     return real_termination_times, real_deadlines
#
# def build_input_from_text_file(file_name):
#     print(file_name)
#     if len(file_name) < 5 or file_name[-4:] !=  ".txt":
#         file_name += ".txt"
#     file = open(file_name)
#     try:
#         lines = file.readlines()
#         durations_and_deadlines = [int(dur) for dur in re.findall(r'\d+', lines[0])]
#         base_level_actions = []
#         for i in range(0, len(durations_and_deadlines), 2):
#             base_level_actions.append(BaseLevelAction(durations_and_deadlines[i], durations_and_deadlines[i+1]))
#         processes_lines = []
#         for i in range(len(lines)-1, -1, -1):
#             if lines[i][0] in ['m', 'd', 'H']:
#                 processes_lines.insert(0, lines[i])
#         processes = []
#         for i in range(0, len(processes_lines), 3):
#             m = build_dist_from_line(processes_lines[i])
#             d = build_dist_from_line(processes_lines[i+1])
#             H = [base_level_actions[int(index)] for index in re.findall(r'\d+', processes_lines[i+2])]
#             processes.append(Process(m, d, H))
#     finally:
#         file.close()
#     return processes, base_level_actions

def build_input_from_text_file(file_name):
    print(file_name)
    if len(file_name) < 5 or file_name[-4:] !=  ".txt":
        file_name += ".txt"
    file = open(file_name)
    try:
        lines = file.readlines()
        durations_and_deadlines = [int(dur) for dur in re.findall(r'\d+', lines[0])]
        base_level_actions = []
        for i in range(0, len(durations_and_deadlines), 2):
            base_level_actions.append(BaseLevelAction(durations_and_deadlines[i], durations_and_deadlines[i+1]))
        real_termination_times_lists = []
        real_deadlines_lists = []
        i = 2
        while lines[i][0] == 'r':
            real_termination_times_lists.append([int(tt) for tt in re.findall(r'\d+', lines[i])[1:]])
            real_deadlines_lists.append([int(dl) for dl in re.findall(r'\d+', lines[i + 1])[1:]])
            i += 3
        processes_lines = []
        for i in range(len(lines)-1, -1, -1):
            if lines[i][0] in ['m', 'd', 'H']:
                processes_lines.insert(0, lines[i])
        processes = []
        for i in range(0, len(processes_lines), 3):
            m = build_dist_from_line(processes_lines[i])
            d = build_dist_from_line(processes_lines[i+1])
            H = [base_level_actions[int(index)] for index in re.findall(r'\d+', processes_lines[i+2])]
            processes.append(Process(m, d, H))
    finally:
        file.close()
    return processes, base_level_actions, real_termination_times_lists, real_deadlines_lists

def fail_processes_due_to_deadlines_of_blas(ns, processes):
    for i, p in enumerate(processes):
        earliest_time_to_done_exec = ns[0] + ns[2]
        for b in p.H[ns[3]:]:
            earliest_time_to_done_exec += b.duration
            if earliest_time_to_done_exec > b.deadline:
                ns[1][i] = -math.inf
                break

def compute_sum_of_durations(l, p):
    sum_durs = 0
    for l in range(l, len(p.H)):
        sum_durs += p.H[l].duration
    return sum_durs

def run_heuristic_on_input(processes, base_level_actions, real_deadlines, real_termination_times, hnap, heuristic=None):
    n = len(processes)
    n_minus_infs = [-math.inf] * n
    s = [0, [0]*n, 0, 0]
    runtime = time.time()
    if heuristic is None:
        heuristic = build_heuristic(hnap[0], processes, base_level_actions, hnap[1:])
    while True:
        action = heuristic.get_next_action(s)
        # print(action)
        if action is None or s[1] == n_minus_infs:
            s = "FAIL"
            break
        if isinstance(action, int):
            if s[1][action] == -math.inf:
                s = "FAIL"
                break
            s[0] += 1
            s[1][action] += 1
            s[2] = max(0, s[2]-1)
            fail_processes_due_to_deadlines_of_blas(s, processes)
            if s[1][action] != -math.inf and real_termination_times[action] == s[1][action]:
                # above it was checked that the partial plan is still applicable for the maximal optional deadline of the process, and
                # here it is checked for the real deadline of the process
                p_i = processes[action]
                sum_durs = compute_sum_of_durations(s[3], p_i)
                if s[0] + s[2] + sum_durs <= real_deadlines[action]:
                    s = "SUCCESS"
                    break
                else:
                    s[1][action] = -math.inf
        else:
            if s[2] > 0:
                s = "FAIL"
                break
            s[2] = action.duration
            s[3] += 1
            for i in range(0, n):
                if s[3] > len(processes[i].H) or not processes[i].H[s[3]-1].serial_number == action.serial_number:
                    s[1][i] = -math.inf
    runtime = time.time() - runtime
    return s, runtime

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
        return "rmlet" + "+" + shortened_name([heuristic_full_name_and_params[1][0]])
    if heuristic_full_name == "Laziness":
        return "mlet" + "+" + shortened_name([heuristic_full_name_and_params[1]])
    if heuristic_full_name == "MCTS":
        return "mcts" + "+" + str(heuristic_full_name_and_params[1])
    if heuristic_full_name == "KBounded":
        return str(heuristic_full_name_and_params[1]) + "-bnd" + "+" + shortened_name([heuristic_full_name_and_params[2]])

def evaluate_all_heuristics_empirically(folder, inputs, he_names_and_params, kd=False):
    all_heuristics = [shortened_name(hnap) for hnap in he_names_and_params]
    for i, input in enumerate(inputs):
        Process.reset_serial_number()
        BaseLevelAction.reset_serial_number()
        processes, base_level_actions, real_termination_times_lists, real_deadlines_lists = build_input_from_text_file(folder + "/" + input)
        times_to_run_any_input = len(real_termination_times_lists)
        number_of_heuristics = len(he_names_and_params)
        runtime_averages = [[0.0] * number_of_heuristics for _ in range(times_to_run_any_input)]
        success_ratios = [[0.0] * number_of_heuristics for _ in range(times_to_run_any_input)]
        if kd:
            from_ud_to_kd_deadlines(processes)
            real_deadlines_lists = [[p.d.dist_list[0][1] for p in processes] for _ in range(times_to_run_any_input)]
        for j in range(times_to_run_any_input):
            for k, hnap in enumerate(he_names_and_params):
                print(hnap[0])
                last_state, runtime = run_heuristic_on_input(deepcopy(processes), base_level_actions, real_deadlines_lists[j], real_termination_times_lists[j], hnap)
                runtime_averages[j][k] += runtime
                if last_state == "SUCCESS":
                    success_ratios[j][k] += 1
        results_file = open("15-puzzle-multiple-sampling/results_{}/{}".format(len(processes), input), 'w')
        for j in range(len(all_heuristics)):
            result = all_heuristics[j] + ": " + str(sum([ratios[j] for ratios in success_ratios]) / times_to_run_any_input) + ", " + \
                  str(sum([runtimes[j] / times_to_run_any_input for runtimes in runtime_averages]))[:4]
            results_file.write(result + "\n")
            print(result)
        results_file.close()


def evaluate_all_heuristics_probabilistically_kd(folder, inputs, he_names_and_params):
    all_heuristics = [shortened_name(hnap) for hnap in he_names_and_params]
    for i, input in enumerate(inputs):
        print("evaluating input {}".format(i))
        Process.reset_serial_number()
        BaseLevelAction.reset_serial_number()
        processes, base_level_actions, real_termination_times_lists, real_deadlines_lists = build_input_from_text_file(folder + "/" + input)
        from_ud_to_kd_deadlines(processes)
        results_file = open("15-puzzle-multiple-sampling/results_{}/{}".format(len(processes), input), 'a')
        for j, hnap in enumerate(he_names_and_params):
            print(hnap[0])
            heuristic = build_heuristic(hnap[0], processes, base_level_actions, hnap[1:])
            prob = evaluate_heuristic_known_deadlines(processes, heuristic)
            results_file.write(all_heuristics[j] + ": " + str(prob) + "\n")
            print(all_heuristics[j] + ": " + str(prob))
        results_file.close()


def evaluate_all_heuristics_probabilistically_efficiently(folder, inputs, he_names_and_params, kd=False):
    all_heuristics = [shortened_name(hnap) for hnap in he_names_and_params]
    for i, input in enumerate(inputs):
        print("evaluating input {}".format(i))
        Process.reset_serial_number()
        BaseLevelAction.reset_serial_number()
        processes, base_level_actions, real_termination_times_lists, real_deadlines_lists = build_input_from_text_file(folder + "/" + input)
        times_to_run_any_input = len(real_termination_times_lists)
        number_of_heuristics = len(he_names_and_params)
        runtime_averages = [[0.0] * number_of_heuristics for _ in range(times_to_run_any_input)]
        success_ratios = [[0.0] * number_of_heuristics for _ in range(times_to_run_any_input)]
        if kd:
            from_ud_to_kd_deadlines(processes)
        for j, hnap in enumerate(he_names_and_params):
            print(hnap[0])
            heuristic = None
            real_runtime = -1
            if hnap[0] in ['MDP']:
                real_runtime = time.time()
                heuristic = build_heuristic(hnap[0], processes, base_level_actions, hnap[1:])
                real_runtime = time.time() - real_runtime
            for k in range(times_to_run_any_input):
                last_state, runtime = run_heuristic_on_input(deepcopy(processes), base_level_actions,
                                        real_deadlines_lists[k], real_termination_times_lists[k],hnap, heuristic)
                if real_runtime != -1:
                    runtime = real_runtime
                runtime_averages[k][j] += runtime
                if last_state == "SUCCESS":
                    success_ratios[k][j] += 1
        results_file = open("15-puzzle-multiple-sampling/results_{}/{}".format(len(processes), input), 'a')
        for j in range(len(all_heuristics)):
            result = all_heuristics[j] + ": " + str(sum([ratios[j] for ratios in success_ratios]) / times_to_run_any_input) + ", " + \
                     str(sum([runtimes[j] / times_to_run_any_input for runtimes in runtime_averages]))[:4]
            results_file.write(result + "\n")
            print(result)
        results_file.close()


def run_tests(kd=False):
    folder = "15-puzzle-multiple-sampling/helper_folder"
    inputs = os.listdir(folder)
    # inputs = ['input21']
    # inputs = inputs[22:31]
    he_names_and_params = [
        ["MDP", None, None],
        ["SAE2DynamicProgramming", None],
        ["Random", None],
        ["MostPromisingPlan", None],
        ["ExecutiveMostPromisingPlan", None],
        ["RoundRobin", None],
        ["ExecutiveRoundRobin", None],
        ["BasicGreedyScheme", 0],
        ["ExecutiveBasicGreedyScheme", 0],
        ["DDA", 1, 1],
        ["ExecutiveDDA", 1, 1],
        # ["DDAOnMaxSize", 1, 1],
        # ["JustDoSomething", ["SAE2DynamicProgramming", 1, None]],
        ["JustDoSomething", ["BasicGreedyScheme", 1, 0]],
        # ["JustDoSomething", ["DDA", 1, 1, 1]],
        ["Laziness", "SAE2DynamicProgramming", None],
        ["Laziness", "BasicGreedyScheme", 0],
        ["Laziness", "DDA", 1, 1],
        ["MCTS", 10],
        # ["MCTS", 50],
        ["MCTS", 100],
        # ["MCTS", 200],
        ["MCTS", 500],
        # ["MCTS", 1000],
        # ["KBounded", 2, "SAE2DynamicProgramming", [None]],
        ["KBounded", 2, "BasicGreedyScheme", [0]],
        # ["KBounded", 2, "DDA", [1, 1]],
        # ["KBounded", 3, "SAE2DynamicProgramming", [None]],
        # ["KBounded", 3, "BasicGreedyScheme", [0]],
        # ["KBounded", 3, "DDA", [1, 1]],
    ]
    # evaluate_all_heuristics_empirically(folder, inputs, he_names_and_params, kd)
    # evaluate_all_heuristics_probabilistically_kd(folder, inputs, he_names_and_params)
    evaluate_all_heuristics_probabilistically_efficiently(folder, inputs, he_names_and_params, kd)


if __name__ == "__main__":
    kd = False
    # kd = True
    run_tests(kd)