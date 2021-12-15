import random
import re

from ipae_mdp import BaseLevelAction
from ipae_mdp import Distribution
from ipae_mdp import Process


def __build_dist_from_line(line):
    '''parse a string and create a PMF'''
    line_numbers = [float(f) for f in re.findall(r'\d+(?:\.\d+)?', line)]
    dist_list = []
    for i in range(0, len(line_numbers), 2):
        dist_list.append((line_numbers[i], int(line_numbers[i+1])))
    return Distribution(dist_list)

def build_input_from_text_file(file_name):
    '''
    Given a file name, this function builds lists of Processes and BaseLevelActions, which constitute the input for IPAE.
    The file should be a text file and located in folder "ipae_inputs".
    The function does not deal with input legality but just parse the file. In particular, it is assumed that the last optional
    deadline is not after the last possible termination time.
    The first line in the file should contain the durations of the base-level actions (order matters for the serial numbers).
    The processes should be written in three lines, one line for any field of Process - m, d and H (in this order).
    The lines for m, d and H must start with the letters 'm', 'd' and 'H', and no other line can start with these letters.
    The lines of a process must be written consecutively among those that start with 'm', 'd' or 'H'. Any other line is ignored.
    The base-level actions should be a sequence of integers. The distributions should be sequence of pairs (in some representation -
    just numbers are not ignored). All the numbers must be non-negative.
    '''
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
        processes_lines = []
        for i in range(len(lines)-1, -1, -1):
            if lines[i][0] in ['m', 'd', 'H']:
                processes_lines.insert(0, lines[i])
        processes = []
        for i in range(0, len(processes_lines), 3):
            m = __build_dist_from_line(processes_lines[i])
            d = __build_dist_from_line(processes_lines[i+1])
            H = [base_level_actions[int(index)] for index in re.findall(r'\d+', processes_lines[i+2])]
            processes.append(Process(m, d, H))
    finally:
        file.close()
    return processes, base_level_actions


def __create_dist(maximal_deadline, m_dist=False, cur_max_deadline=None, known_deadlines=False):
    if cur_max_deadline is None:
        cur_max_deadline = random.randrange(1, maximal_deadline + 1)
    if known_deadlines:
        list_len = 1
    else:
        list_len = random.randrange(1, min(cur_max_deadline + 1, 20))
    times = random.sample(range(1, cur_max_deadline + 1), list_len)
    times.sort()
    if not m_dist and times[-1] < maximal_deadline // 4: # make sure that the deadline is not too early
        max_d = random.randint(float(maximal_deadline // 4), maximal_deadline + 1)
        if known_deadlines:
            times[0] = max_d
        else:
            times.append(max_d)
    if m_dist: # the maximal termination time is assumed to be bigger than the maximal deadline
        times.append(maximal_deadline + 1)
    # any probability gets some number in [1,999], and the numbers are normalized by their sum to get a distribution
    probs = [random.randrange(1, 1000) for _ in range(len(times))]
    normalization_factor = 0
    for num in probs:
        normalization_factor += num
    for j in range(len(times)):
        probs[j] /= normalization_factor
    return [(probs[j], times[j]) for j in range(len(times))] # pairing probs and times

def __dist_to_string(dist):
    ret = "["
    for (p, t) in dist:
        ret += str(p) + " : " + str(t) + ", "
    return ret[:-2] + "]"

def __create_H_list(blas, cur_max_deadline):
    hl_len = random.randrange(len(blas))
    hl_indices = random.sample(range(len(blas)), hl_len)
    # fix the partial plan in case that it is initially too long to meet the deadline or there are actions which can't meet their
    # deadlines
    sum_durs = 0
    i = 0
    while i < len(hl_indices):
        b = hl_indices[i]
        sum_durs += blas[b][0]
        if sum_durs > cur_max_deadline or sum_durs > blas[b][1]:
            del hl_indices[i]
        else:
            i += 1
    return hl_indices

def create_input(folder, input_name, number_of_blas, bla_max_dur, number_of_processes, maximal_deadline, known_deadlines=False):
    file = open(folder + '/' + input_name + ".txt", 'w')
    try:
        file.write("base-level actions:\t")
        blas_durs = [random.randrange(1, bla_max_dur) for _ in range(number_of_blas)]
        # deadlines of base-level actions are all pretty late (in [0.5 * maximal_deadline, maximal_deadline]) to enlarge the
        # probability of longer partial plans to be valid
        blas = [(blas_durs[i], random.randrange(max(blas_durs[i], 0.5 * maximal_deadline), maximal_deadline))
                                                                                    for i in range(number_of_blas)]
        file.write(str(blas) + "\n\n")
        for i in range(number_of_processes):
            d_list = __create_dist(maximal_deadline, known_deadlines=known_deadlines)
            cur_max_deadline = d_list[-1][1]
            m_list = __create_dist(maximal_deadline, m_dist=True, cur_max_deadline=cur_max_deadline)
            H_list = __create_H_list(blas, cur_max_deadline)
            file.write("p" + str(i) + ":\t\n")
            file.write("m:\t" + __dist_to_string(m_list) + "\n")
            file.write("d:\t" + __dist_to_string(d_list) + "\n")
            file.write("H:\t" + str(H_list) + "\n\n")
    finally:
        file.close()



# folder = "ipae_input_bank/5_5_5_50_k_20"
# num_of_blas = 5
# bla_max_dur = 5
# num_of_procs = 5
# maximal_deadline = 50
# kd = True
# inputs = 20
# for i in range(0,inputs):
#     create_input(folder, "input" + str(i), num_of_blas, bla_max_dur, num_of_procs, maximal_deadline, known_deadlines=kd)