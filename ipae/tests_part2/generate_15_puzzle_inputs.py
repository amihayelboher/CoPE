import os
import random
import ast
from copy import deepcopy
import math

from tile_puzzle_15_a_star import manhattan_distance
from tile_puzzle_15_a_star import create_goal_puzzle_and_locations_dict
from tile_puzzle_15_a_star import play

from ipae_mdp_demo import Distribution


def compress_ml_values(ml, backets):
    minimal_value = ml[0][1]
    maximal_value = ml[-1][1]
    backet_size = (maximal_value - minimal_value) / backets
    offset = minimal_value
    new_ml = []
    ml_index = 0
    for i in range(1, backets + 1):
        prob = 0.0
        backet_time = offset + i * backet_size
        while ml_index < len(ml) and ml[ml_index][1] <= backet_time:
            prob += ml[ml_index][0]
            ml_index += 1
        if prob != 0.0:
            new_ml.append((prob, int(backet_time)))
    return new_ml

def compress_ml_probabilities(ml, backets):
    prob = 1 / backets
    new_ml = []
    ml_index = 0
    while ml_index < len(ml):
        acc_prob = 0.0
        value = 0
        while ml_index < len(ml) and acc_prob + ml[ml_index][0] <= prob:
            acc_prob += ml[ml_index][0]
            value = ml[ml_index][1]
            ml_index += 1
        if acc_prob != 0.0:
            new_ml.append((acc_prob, value))
        else:
            new_ml.append(ml[ml_index])
            ml_index += 1
    return new_ml

def backeting(m_lists, backets, criterion=None):
    compressed_m_lists = m_lists
    if criterion == "values":
        compressed_m_lists = [compress_ml_values(ml, backets) for ml in m_lists]
    if criterion == "probabilities":
        compressed_m_lists = [compress_ml_probabilities(ml, backets) for ml in m_lists]
    return compressed_m_lists

def read_lists(file_name, backets=0, criterion=None):
    lists = None
    file = open(file_name, 'r')
    try:
        lists = [ast.literal_eval(line) for line in file.readlines()]
        if backets > 0:
            lists = backeting(lists, backets, criterion)
    finally:
        file.close()
    return lists

def get_nodes_from_a_star(num_of_processes_in_input, relative=False):
    # fringe = play(fringe_size=num_of_processes_in_input)
    # return random.sample(fringe, num_of_processes_in_input)
    return play(fringe_size=num_of_processes_in_input, relative=relative)

def zero_indices(pos):
    for i in range(4):
        for j in range(4):
            if pos[i][j] == 0:
                return i, j

def find_move(state1, state2):
    i1, j1 = zero_indices(state1)
    i2, j2 = zero_indices(state2)
    if i1 > i2:
        return 0
    if i1 < i2:
        return 1
    if j1 > j2:
        return 2
    return 3

def build_H(node):
    H = []
    while node[1]:
        move = find_move(node[1][-1], node[-1])
        H.insert(0, move)
        node = node[1]
    return H

def get_deadline(H, m):
    prob = random.random()
    minimal_bound = len(H)
    maximal_bound = -1
    M = Distribution(m).PMF_to_CDF()
    for p, t in M.dist_list:
        if p > prob:
            maximal_bound = t
            break
        minimal_bound = t + 1
    return random.randint(minimal_bound, maximal_bound)

def compact_m_by_ratio(m, ratio):
    m = [(p, math.ceil(t / ratio)) for (p,t) in m]
    compact_m = []
    i = 0
    while i < len(m):
        cur_t = m[i][1]
        cur_p = 0.0
        while i < len(m) and m[i][1] == cur_t:
            cur_p += m[i][0]
            i += 1
        compact_m.append((cur_p, cur_t))
    return compact_m

def create_inputs(folder, num_of_inputs, num_of_processes_in_input, execution_time_of_bla, num_of_samplings, backets=0,
                  criterion=None, relative=False, func=None):
    for i in range(num_of_inputs):
        file_name = folder + "/input" + str(i) + ".txt"
        file = open(file_name, 'w')
        try:
            # blas = [(execution_time_of_bla, 1000000) for _ in range(4)]
            blas = [(1, 1000000) for _ in range(4)]
            file.write("base-level actions:\t")
            file.write(str(blas) + "\n\n")
            _, goal_locations = create_goal_puzzle_and_locations_dict()
            nodes = get_nodes_from_a_star(num_of_processes_in_input, relative)
            max_permitted_length = 50
            if relative:
                max_permitted_length, nodes = nodes[0], nodes[1]
                # max_permitted_length = math.floor(max_permitted_length + func(max_permitted_length))
                max_permitted_length = math.floor(func(max_permitted_length))
            m_lists = read_lists("m_lists_part2.txt", backets, criterion)
            reversed_r_lists = read_lists("r_lists_part2.txt", backets)
            for r in reversed_r_lists:
                r.reverse()
            ms, ds, Hs = [], [], []
            for node in nodes:
                dist = manhattan_distance(node[-1], goal_locations)
                H = build_H(node)
                Hs.append(H)
                closest_bigger_even = ((dist + 1) // 2) * 2
                # ds.append(Distribution([(1, random.randint(len(H), m_lists[closest_bigger_even][-1][1] - 1))]))
                # ds.append(Distribution([(1, get_deadline(H, m))]))
                r = reversed_r_lists[closest_bigger_even]
                # d_minus_r = [(pair[0], max_permitted_length * execution_time_of_bla - pair[1]) for pair in r]
                d_minus_r = [(pair[0], max_permitted_length - pair[1]) for pair in r]
                ds.append(Distribution(d_minus_r))
                m = deepcopy(m_lists[closest_bigger_even])
                m = compact_m_by_ratio(m, execution_time_of_bla)
                latest_valid_at = d_minus_r[-1][1]
                acc_probs = 0.0
                last_index = 0
                for _, t in m:
                    if t > latest_valid_at:
                        break
                    acc_probs += m[last_index][0]
                    last_index += 1
                m = m[:last_index + 1]
                if len(m) != 1:
                    m[-1] = (1 - acc_probs, latest_valid_at + 1)
                m = compress_ml_values(m, backets)
                if last_index == 0:
                    m.append((0, latest_valid_at + 2))
                if len(m) == 1:
                    print(i, m)
                ms.append(Distribution(m))
            for i in range(num_of_samplings):
                real_termination_times = [m.sample() for m in ms]
                real_deadlines = [d.sample() for d in ds]
                file.write("real termination times " + str(i) + ":	" + str(real_termination_times) + "\n")
                file.write("real deadlines " + str(i) + ": " + str(real_deadlines) + "\n\n")
            for j in range(len(nodes)):
                file.write("p" + str(j) + ":\t\n")
                file.write("m:\t" + str(ms[j]) + "\n")
                file.write("d:\t" + str(ds[j]) + "\n")
                file.write("H:\t" + str(Hs[j]) + "\n\n")
        finally:
            file.close()


if __name__ == "__main__":
    num_of_inputs = 25
    num_of_samplings = 1
    num_of_processes_in_input = 5
    execution_time_of_bla = 1
    backets = 10
    criterion = None
    relative = True
    # func = lambda x : int(2.5 * x)
    # func = lambda x : int(math.log2(num_of_processes_in_input) * x)
    func = lambda x : int(4 * x)
    folder = "15-puzzle-multiple-sampling/" + str(num_of_inputs) + "_" + str(num_of_processes_in_input) + "_" + str(execution_time_of_bla)
    folder = "15-puzzle-multiple-sampling/helper_folder"
    create_inputs(folder, num_of_inputs, num_of_processes_in_input, execution_time_of_bla, num_of_samplings,
                  backets=backets, criterion=criterion, relative=relative, func=func)