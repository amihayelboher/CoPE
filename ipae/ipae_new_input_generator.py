import random
import math
import re
from copy import deepcopy

from ipae_mdp import BaseLevelAction
from ipae_mdp import Distribution
from ipae_mdp import Process


def build_dist_from_line(line):
    '''parse a string and create a PMF'''
    line_numbers = [float(f) for f in re.findall(r'\d+(?:\.\d+)?', line)]
    dist_list = []
    for i in range(0, len(line_numbers), 2):
        dist_list.append((line_numbers[i], int(line_numbers[i+1])))
    return Distribution(dist_list)

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
        real_termination_times = None
        real_deadlines = None
        if lines[2][0] == 'r':
            real_termination_times = [int(tt) for tt in re.findall(r'\d+', lines[2])]
            real_deadlines = [int(dl) for dl in re.findall(r'\d+', lines[3])]
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
    return processes, base_level_actions, real_termination_times, real_deadlines


def create_partial_plans_trie(num_of_internal_nodes, num_of_leaves, bla_max_dur):
    dur = random.randint(1, bla_max_dur)
    dl = dur + random.randint(0, int((dur + bla_max_dur) / 2))
    internal_nodes = {0 : {'is_leaf' : False, 'bla' : (dur, dl)}}
    for i in range(1, num_of_internal_nodes + num_of_leaves):
        r = random.randint(0, min(i, num_of_internal_nodes) - 1)
        dur = random.randint(1, bla_max_dur)
        dl = internal_nodes[r]['bla'][1] + dur + random.randint(0, int((dur + bla_max_dur / 2) / 2))
        internal_nodes[i] = {'is_leaf' : i >= num_of_internal_nodes, 'bla' : (dur, dl)}
        internal_nodes[r][i] = internal_nodes[i]
    return internal_nodes[0]

def ppt2pps(ppt):
    def ppt2pps(ppt, blas, pps, acc):
        blas.append(ppt['bla'])
        for k in ppt.keys():
            acc_copy = [b for b in acc]
            if k != 'is_leaf' and k != 'bla':
                acc_copy.append(k - 1)
                ppt2pps(ppt[k], blas, pps, acc_copy)
            elif k == 'is_leaf' and ppt[k]:
                pps.append(acc_copy)
    pps = []
    blas = []
    ppt2pps(ppt, blas, pps, [])
    return blas[:-1], pps

def compute_pps_lengths(blas, pps):
    lengths = []
    for pp in pps:
        l = 0
        for b in pp:
            l += blas[b][0]
        lengths.append(l)
    return lengths

def create_d_list(dur_of_pp, maximal_optional_deadline, known_deadlines=False):
    maximal_deadline = 1
    while maximal_deadline < 3:
        maximal_deadline = random.randint(dur_of_pp, maximal_optional_deadline)
    d_list_len = 1
    if not known_deadlines:
        d_list_len = random.randint(1, min(maximal_deadline, 15))
    times = random.sample(range(1, maximal_deadline), d_list_len - 1)
    times.sort()
    times.append(maximal_deadline)
    probs = [random.randrange(1, 1000) for _ in range(len(times))]
    normalization_factor = sum(probs)
    for j in range(len(times)):
        probs[j] /= normalization_factor
    return [(probs[j], times[j]) for j in range(len(times))]

def create_m_dist(maximal_deadline):
    m_list_len = random.randint(2, min(maximal_deadline - 1, 10))
    times = random.sample(range(1, maximal_deadline), m_list_len - 1)
    times.sort()
    times.append(maximal_deadline + 1)
    probs = [random.randrange(1, 1000) for _ in range(len(times))]
    normalization_factor = 2 * sum(probs)
    for j in range(len(times)):
        probs[j] /= normalization_factor
    probs[-1] += 0.5
    return [(probs[j], times[j]) for j in range(len(times))]

def fix_dls_of_blas(blas, pps):
    acc_exec_time_of_prefix = []
    for pp_i in pps:
        acc_time_of_pp_i = []
        if len(pp_i) > 0:
            acc_time_of_pp_i.append(blas[pp_i[0]][0])
        blas_of_p_i = [blas[i] for i in pp_i[1:]]
        for i, bla in enumerate(blas_of_p_i):
            acc_time_of_pp_i.append(acc_time_of_pp_i[i] + bla[0])
        acc_exec_time_of_prefix.append(acc_time_of_pp_i)
    for i, times in enumerate(acc_exec_time_of_prefix):
        for j, t in enumerate(times):
            if blas[pps[i][j]][1] < t:
                blas[pps[i][j]] = (blas[pps[i][j]][0], t + random.randint(0, blas[pps[i][j]][1]))

def sample_real_times(list_of_lists, are_m_lists=False):
    real_times = []
    for l in list_of_lists:
        l = deepcopy(l)
        if are_m_lists:
            # l[-1] = (l[-1][0]-0.5, l[-1][1])
            l[-1] = (l[-1][0], l[-1][1]-0.5)
            l = [(2*p, n) for (p,n) in l]
        dist = Distribution(l)
        real_times.append(int(dist.sample()))
    return real_times

def dist_to_string(dist):
    ret = "["
    for (p, t) in dist:
        ret += str(p) + " : " + str(t) + ", "
    return ret[:-2] + "]"

def create_input(folder, input_name, bla_max_dur, number_of_processes, known_deadlines=False):
    file = open(folder + '/' + input_name + ".txt", 'w')
    try:
        number_of_pps = int(math.log2(number_of_processes))
        ppt = create_partial_plans_trie(number_of_pps, number_of_processes, bla_max_dur)
        blas, pps = ppt2pps(ppt)
        durs_of_pps = compute_pps_lengths(blas, pps)
        maximal_optional_deadline = max([b[1] for b in blas])
        m_lists = []
        d_lists = []
        for i in range(number_of_processes):
            d_list = create_d_list(durs_of_pps[i], maximal_optional_deadline, known_deadlines=known_deadlines)
            cur_max_deadline = d_list[-1][1]
            m_list = create_m_dist(cur_max_deadline)
            m_lists.append(m_list)
            d_lists.append(d_list)
        fix_dls_of_blas(blas, pps)
        file.write("base-level actions:\t")
        file.write(str(blas) + "\n\n")
        real_termination_times = sample_real_times(m_lists, are_m_lists=True)
        real_deadlines = sample_real_times(d_lists)
        file.write("real termination times:\t" + str(real_termination_times) + "\n")
        file.write("real deadlines:\t" + str(real_deadlines) + "\n\n")
        for i in range(number_of_processes):
            file.write("p" + str(i) + ":\t\n")
            file.write("m:\t" + dist_to_string(m_lists[i]) + "\n")
            file.write("d:\t" + dist_to_string(d_lists[i]) + "\n")
            file.write("H:\t" + str(pps[i]) + "\n\n")
    finally:
        file.close()



if __name__ == "__main__":
    bla_max_dur = 5
    num_of_procs = 30
    kd = True
    inputs = 10
    kd_or_ud_str = 'kd' if kd else 'ud'
    folder = "ipae_input_bank/" + str(bla_max_dur) + "_" + str(num_of_procs) + "_" + kd_or_ud_str  + "_" + str(inputs)
    folder = "ipae_input_bank/test_generator"
    # print(folder)
    for i in range(0,inputs):
        create_input(folder, "input" + str(i), bla_max_dur, num_of_procs, known_deadlines=kd)