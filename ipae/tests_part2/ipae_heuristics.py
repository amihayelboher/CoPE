from abc import ABC, abstractmethod
from copy import deepcopy
import math
import random
import time
import itertools

from mdp import MDP2
from mdp import best_policy
from mdp import value_iteration
from mdp import policy_iteration

from ipae_mdp import Process
from ipae_mdp import BaseLevelAction
from ipae_mdp import Distribution

# from ipae_input_generator import build_input_from_text_file as old_input_builder

from ipae_new_input_generator import build_input_from_text_file


def compute_sum_of_durations(l, p):
    sum_durs = 0
    for l in range(l, len(p.H)):
        sum_durs += p.H[l].duration
    return sum_durs

def copmute_execution_blas_times_list(process, deadline_of_process):
    '''compute the lazy execution time of any of the actions in the partial plan'''
    execution_blas_times_list = []
    cur_max_exec_time = deadline_of_process
    for b in reversed(process.H):
        cur_max_exec_time = min(cur_max_exec_time, b.deadline) - b.duration
        execution_blas_times_list.insert(0, cur_max_exec_time)
    return execution_blas_times_list

def __compute_next_ns_and_p_success(processes, s, a, n, known_deadlines, n_minus_infs):
    '''
    compute the next ns and success probability in the case of known deadlines, where if the next ns might be some non-terminal
    ns (with a strict positive probability) or SUCCESS, the returned ns is the non-terminal ns
    '''
    prob = 0.0
    if s[1] == n_minus_infs:
        return "FAIL", 0.0
    if isinstance(a, int): # a is a computation action
        if s[1][a] == -math.inf:
            return "FAIL", 0.0
        ns = deepcopy(s)
        ns[0] += 1
        ns[1][a] += 1
        ns[2] = max(0, s[2]-1)
        # check the deadline of the process
        p_i = processes[a]
        sum_durs = compute_sum_of_durations(s[3], p_i)
        # by simple algebraic transitions, the next condition is equivalent to:
        # T + 1 > f_i(s) + W - max(0, W-1) = f_i(ns) = d_i - dur(H_i[L+1...|H_i|]) - max(0, W-1)
        if ns[0] + sum_durs + ns[2] > known_deadlines[a]:
            return "FAIL", 0.0
        # check the deadline of the first base-level actions in the partial plan (if OK, any of the next is also OK)
        execution_blas_times_list = copmute_execution_blas_times_list(p_i, p_i.d.dist_list[0][1])
        if len(p_i.H) > s[3] and execution_blas_times_list[s[3]] == s[0]:
            ns[1][a] = -math.inf
            return ns, 0.0
        # else, no deadline is violated - neither of the process nor of the base-level actions
        p_Ci_s_ns = p_i.m.compute_probability(s[1][a]+1) / (1 - p_i.M.compute_probability(s[1][a]))
        if p_Ci_s_ns == 1:
            return "SUCCESS", 1.0
        return ns, p_Ci_s_ns
    else: # a is a base-level action
        if s[2] > 0:
            return "FAIL", 0.0
        s[2] = a.duration
        s[3] += 1
        for i in range(0,n):
            if s[3] > len(processes[i].H) or not processes[i].H[s[3]-1] == a:
                s[1][i] = -math.inf
    return s, prob


def evaluate_schedule_known_deadlines(processes, schedule):
    '''
    compute the probability of finding a timely solution by executing the actions in the given schedule in order in the case
    of known deadlines
    '''
    n = len(processes)
    known_deadlines = [p.D.dist_list[-1][1] for p in processes]
    n_minus_infs = [-math.inf] * n
    prob = 0.0
    cur_s = [0, [0]*n, 0, 0] # starting from initial ns
    index = 0
    while index < len(schedule): # assume that if all the actions in schedule were executed and didn't reach non-terminal ns
                                 # then it immediately fails
        cur_s, p_success = __compute_next_ns_and_p_success(processes, cur_s, schedule[index], n, known_deadlines, n_minus_infs)
        if cur_s == "SUCCESS":
            return 1.0
        elif cur_s == "FAIL":
            return prob
        else:
            prob += (1 - prob) * p_success # either already succeeded or succeed now given that didn't succeed yet
            index += 1
    return prob


def evaluate_heuristic_known_deadlines(processes, heuristic):
    '''
    compute the probability of finding a timely solution by executing actions according to a heuristic in the known deadlines
    case
    '''
    n = len(processes)
    known_deadlines = [p.D.dist_list[-1][1] for p in processes]
    n_minus_infs = [-math.inf] * n
    prob = 0.0
    cur_s = [0, [0]*n, 0, 0] # starting from initial ns
    while True:
        action = heuristic.get_next_action(cur_s)
        if action is None:
            return prob
        cur_s, p_success = __compute_next_ns_and_p_success(processes, cur_s, action, n, known_deadlines, n_minus_infs)
        if cur_s == "SUCCESS":
            return 1.0
        elif cur_s == "FAIL":
            return prob
        prob += (1 - prob) * p_success # either already succeeded or succeed now given that didn't succeed yet


def fail_processes_due_to_deadlines_of_blas(ns, processes):
    for i, p in enumerate(processes):
        earliest_time_to_done_exec = ns[0] + ns[2]
        for b in p.H[ns[3]:]:
            earliest_time_to_done_exec += b.duration
            if earliest_time_to_done_exec > b.deadline:
                ns[1][i] = -math.inf
                break

def __run_heuristic_on_input(processes, base_level_actions, real_deadlines, real_termination_times, hnap):
    n = len(processes)
    n_minus_infs = [-math.inf] * n
    s = [0, [0]*n, 0, 0]
    runtime = time.time()
    heuristic = build_heuristic(hnap[0], processes, base_level_actions, hnap[1:])
    while True:
        action = heuristic.get_next_action(s)
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
                if s[3] > len(processes[i].H) or not processes[i].H[s[3]-1] == action:
                    s[1][i] = -math.inf
    runtime = time.time() - runtime
    return s, runtime


def evaluate_all_heuristics_empirically(folder, inputs, runs, he_names_and_params):
    '''
    evaluate the success and average runtime of all the heuristics on a list of inputs by running them number of times given by the
    user for any of the heuristics
    '''
    number_of_heuristics = len(he_names_and_params)
    runtime_averages = [[0.0] * number_of_heuristics for _ in range(len(inputs))]
    success_ratios = [[0.0] * number_of_heuristics for _ in range(len(inputs))]
    for i, input in enumerate(inputs):
        Process.reset_serial_number()
        BaseLevelAction.reset_serial_number()
        # processes, base_level_actions = build_input_from_text_file(folder + "/" + input)
        processes, base_level_actions, real_termination_times, real_deadlines = build_input_from_text_file(folder + "/" + input)
        if real_deadlines is not None:
            runs = 1
        for _ in range(runs):
            if real_deadlines is None:
                real_deadlines = [p.d.sample() for p in processes]
                real_termination_times = [p.m.sample() for p in processes]
            for j, hnap in enumerate(he_names_and_params):
                print(hnap[0])
                last_state, runtime = __run_heuristic_on_input(processes, base_level_actions, real_deadlines,
                                                               real_termination_times, hnap)
                runtime_averages[i][j] += runtime
                if last_state == "SUCCESS":
                    success_ratios[i][j] += 1
        for j in range(number_of_heuristics):
            runtime_averages[i][j] /= runs
            success_ratios[i][j] /= runs
    return runtime_averages, success_ratios


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


def evaluate_one_heuristic_empirically(inputs, runs, heuristic_name, additional_params=None):
    '''
    evaluate the success and average runtime of a heuristic on a list of inputs by running them number of times given by the user
    '''
    runtime_averages = [0.0] * len(inputs)
    success_ratios = [0.0] * len(inputs)
    for i, input in enumerate(inputs):
        Process.reset_serial_number()
        BaseLevelAction.reset_serial_number()
        # processes, base_level_actions = old_input_builder("ipae_input_bank/" + input)
        processes, base_level_actions, real_termination_times, real_deadlines = \
                                                                        build_input_from_text_file("ipae_input_bank/" + input)
        got_deadlines_and_termination_times = False
        if real_deadlines is not None:
            got_deadlines_and_termination_times = True
        for _ in range(runs):
            if not got_deadlines_and_termination_times:
                real_deadlines = [p.d.sample() for p in processes]
                real_termination_times = [p.m.sample() for p in processes]
            last_state, runtime = __run_heuristic_on_input(processes, base_level_actions, real_deadlines,
                                                           real_termination_times, (heuristic_name, additional_params))
            runtime_averages[i] += runtime
            if last_state == "SUCCESS":
                success_ratios[i] += 1
        runtime_averages[i] /= runs
        success_ratios[i] /= runs
    print("average runtimes:", runtime_averages)
    print("success ratios:", success_ratios)


class Heuristic(ABC):
    '''This interface represents a heuristic for the IPAE problem.'''
    
    @abstractmethod
    def get_next_action(self, state):
        '''returns the next action to execute according to the heuristic'''
        pass


class IPAEMDP(MDP2):
    '''This class represents MDP for the IPAE problem according to the assumptions mentioned above.'''
    
    def __init__(self, processes, base_level_actions, gamma=0.9):
        self.processes = processes                              # list of the Processes
        self.init = self.State(0, [0]*len(processes), 0, 0)     # initial state
        self.__B = base_level_actions                           # list of the BaseLevelActions
        self.__C = [i for i in range(0,len(processes))]         # list of the computation actions
        self.A = []                                             # all actions
        self.A.extend(self.__B)
        self.A.extend(self.__C)
        self.__SUCCESS = self.State(math.inf, [0]*len(self.processes), 0, 0)
        self.__FAIL = self.State(-math.inf, [0]*len(self.processes), 0, 0)
        self.states = self.__all_states()                       # set of all the optional States
        self.__states_dict = self.__states_to_dict(self.states)
        self.transitions = self.__all_transitions()             # transition function as a hash table: for any (state, action) pair
                                                                # the transition function keeps a list of (probability, next_state)
                                                                # pairs, where if a state does not appear in the next_state options,
                                                                # its probability to be the next state is 0
        self.terminals = [self.__SUCCESS, self.__FAIL]
        self.rewards = self.__all_rewards()                     # reward function
        # MDP.__init__(self, self.init, self.A, self.terminals, self.transitions, self.rewards, self.states, gamma)
        MDP2.__init__(self, self.init, self.A, self.terminals, self.transitions, self.rewards, gamma)

    def __compute_next_states(self, st, a, n, deadlines, n_minus_infs):
        '''compute all the states that may be resulted by taking action a from state st'''
        if str(st) == str(self.__SUCCESS) or str(st) == str(self.__FAIL): # st is a deepcopy of some state, then the comparison uses
                                                                          # strings
            return []
        if isinstance(a, int): # a is a computation action
            if st.state[0] >= deadlines[a] or st.state[1][a] == -math.inf:
                return []
            st.state[0] += 1
            st.state[1][a] += 1
            st.state[2] = max(0, st.state[2] - 1)
            for i, p in enumerate(self.processes): # check whether the partial plan is still applicable for any of the processes
                earliest_time_to_done_exec = st.state[0] + st.state[2]
                for b in p.H[st.state[3]:]:
                    earliest_time_to_done_exec += b.duration
                    if earliest_time_to_done_exec > b.deadline:
                        st.state[1][i] = -math.inf
                        break
            if st.state[1][a] == -math.inf: # next part is redundant if the process is failed
                return [st]
            sts = [st]
            # the process may fail only in case that the current time has a positive probability in the m distribution
            termination_times = [t for _, t in self.processes[a].m.dist_list]
            if st.state[1][a] in termination_times: # computation action may cause a completion of computation which might fail the process
                st_F = deepcopy(st)
                st_F.state[1][a] = -math.inf
                sts.append(st_F)
            return sts
        else: # a is a base-level action
            if st.state[2] > 0 or st.state[1] == n_minus_infs:
                return []
            st.state[2] = a.duration
            st.state[3] += 1
            for i in range(0,n):
                if st.state[3] > len(self.processes[i].H) or not self.processes[i].H[st.state[3] - 1] == a:
                    st.state[1][i] = -math.inf
            return [st]

    def __all_states(self):
        '''compute all the reachable states in the MDP (and also a few unreachable states)'''
        n = len(self.processes)
        states = set()
        states.add(self.__SUCCESS)
        states.add(self.__FAIL)
        deadlines = [p.D.dist_list[-1][1] for p in self.processes] # dist_list is assumed to be sorted by values (events)
        n_minus_infs = [-math.inf] * n # if [T_1...T_n] are all -infinity then the next state is FAIL independtly on the action taken
        s = self.init
        to_expand = [s]
        to_expand_strs = [str(s)] # equal states may have different pointers and comparison may fail, and comparing with string solves
                                  # the problem
        while len(to_expand) > 0:
            s = to_expand[0]
            del to_expand[0]
            states.add(s)
            for a in self.A:
                next_states = self.__compute_next_states(deepcopy(s), a, n, deadlines, n_minus_infs)
                for next_state in next_states:
                    if not str(next_state) in to_expand_strs:
                        to_expand.append(next_state)
                        to_expand_strs.append(str(next_state))
        return states

    def __states_to_dict(self, states):
        '''
        The functions in mdp.py assume that the State-keys of the transition function are the same as the State-vaules, but when
        building the transition function it is much easier to (deeply) copy the states. To use the same objects as keys and values
        the vaules of the transition function are taken from the dict built here. In other words, when keeping a pair in the list of
        T[s][a], it is wrong to keep a new object, but a new object has the same hash-value as the requested State.
        '''
        states_dict = {}
        for s in states:
            states_dict[hash(s)] = s
        return states_dict

    def __all_transitions(self):
        '''compute the transition function of the MDP'''
        n = len(self.processes)
        deadlines = [p.D.dist_list[-1][1] for p in self.processes]
        n_minus_infs = [-math.inf] * n
        transitions = {}
        for s in self.states:
            transitions[s] = {}
            for a in self.A:
                next_states = self.__compute_next_states(deepcopy(s), a, n, deadlines, n_minus_infs)
                if next_states == []:
                    transitions[s][a] = [(1.0, self.__FAIL)] # do not change self.__FAIL to s or self.__SUCCESS!
                elif isinstance(a, int): # a is a computation action
                    ns = next_states[0]
                    if ns.state[1][a] == -math.inf: # the process cannot meet the deadline of some action in it's partial plan
                        transitions[s][a] = [(1.0, self.__states_dict[hash(ns)])]
                    else:
                        p_i = self.processes[a]
                        sum_durs = 0
                        for l in range(ns.state[3], len(p_i.H)):
                            sum_durs += p_i.H[l].duration
                        q_i_ns = ns.state[0] + ns.state[2] + sum_durs - 1
                        p_Ci_s_ns = p_i.m.compute_probability(ns.state[1][a]) / (1 - p_i.M.compute_probability(s.state[1][a]))
                        p_Fi_ns = p_i.D.compute_probability(q_i_ns)
                        p_Si_ns = 1 - p_Fi_ns
                        pairs = []
                        if 1 - p_Ci_s_ns != 0:
                            pairs.append((1 - p_Ci_s_ns, self.__states_dict[hash(ns)]))
                        if p_Ci_s_ns * p_Fi_ns != 0:
                            pairs.append((p_Ci_s_ns * p_Fi_ns, self.__states_dict[hash(next_states[1])]))
                        if p_Ci_s_ns * p_Si_ns != 0:
                            pairs.append((p_Ci_s_ns * p_Si_ns, self.__SUCCESS))
                        transitions[s][a] = pairs
                else: # a is a base-level action
                    transitions[s][a] = [(1.0, self.__states_dict[hash(next_states[0])])]
        return transitions

    def __all_rewards(self):
        '''compute the rewards for all the states of the MDP'''
        rewards = {}
        for st in self.states:
            if st.state[0] == math.inf: # the reward of SUCCESS is 1
                rewards[st] = 1.0
            else: # the reward of any other state is 0
                rewards[st] = 0.0
        return rewards

    def evaluate_policy_rec(self, policy, state):
        if state == self.__SUCCESS:
            return 1.0
        if state == self.__FAIL:
            return 0.0
        action = policy[self.__states_dict[hash(state)]]
        prob = 0.0
        for (p, s) in self.transitions[state][action]:
            prob += p * self.evaluate_policy_rec(policy, s)
        return prob

    def evaluate_policy(self, policy):
        '''
        Compute the probability of finding a timely solution by following a policy.
        '''
        return self.evaluate_policy_rec(policy, self.__states_dict[hash(self.init)])

    def get_state_object(self, state):
        return self.__states_dict[hash(self.State(state[0], state[1], state[2], state[3]))]


    class State:
        '''This class represents a state in the MDP.'''
        def __init__(self, absolute_time, allocated_times, waiting_time, leading_actions):
            self.state = [absolute_time, allocated_times, waiting_time, leading_actions]
            
        def __str__(self):
            if self.state[0] == math.inf:
                return "SUCCESS"
            if self.state[0] == -math.inf:
                return "FAIL"
            return str(self.state)
        
        def __hash__(self):
            return hash(str(self.state))


class MDPHeuristic(Heuristic):
    '''This class builds an optimal policy using IPAEMDP and returns the relevant action in each state reached during the run.'''

    def __init__(self, processes, base_level_actions, gamma=0.9, use_value_iteration=True):
        self.mdp = IPAEMDP(processes, base_level_actions, gamma)
        # self.states_dict = dict(mdp.states)
        self.policy = best_policy(self.mdp, value_iteration(self.mdp)) if use_value_iteration else policy_iteration(self.mdp)
        print("MDP probability of success:", self.mdp.evaluate_policy(self.policy))

    def get_next_action(self, state):
        return self.policy[self.mdp.get_state_object(state)]


class Schedule(Heuristic):
    '''This class doesn't represent a real heuristic but a schedule, using the interface of a heuristic.'''

    def __init__(self, schedule):
        self.index = 0
        self.schedule = schedule

    def get_next_action(self, state):
        if self.index < len(self.schedule):
            self.index += 1
            return self.schedule[self.index - 1]
        return None # causes an overall failure


def s_i(p, t_i, already_allocated, t_d_i):
    ret = 0
    for t_prime in range(1, t_i + 1):
        ret += p.m.compute_probability(t_prime + already_allocated) * (1 - p.D.compute_probability(t_prime + t_d_i - 1))
    return ret

def LPF_i(p, t_i, already_allocated, t_d_i=0):
    return math.log(1 - s_i(p, t_i, already_allocated, t_d_i))

def dynamic_programming_sae2(processes, dls_and_indices):
    # d_n_minus_1 is the d_n of the paper, because the indices are 0 to n-1, not 1 to n, unlike in the paper
    d_n_minus_1 = dls_and_indices[-1][0]
    # if d_n_minus_1 == 0:
    #     return []
    n = len(processes)
    # table[l][t] holds a pair (value, time-to-allocate) if the already allocated time when reaching to process l is t
    # the second coordinate is used in the backtracking, which we would like to do without LPF calculations and in Theta(n + d_0)
    # table = [[None] * (d_l + 1) for d_l, _ in dls_and_indices]
    table = [[(0.0, 0)] * (d_l + 1) for d_l, _ in dls_and_indices]
    table.append([(0.0, 0)] * (d_n_minus_1 + 1)) # for the case of OPT(., n)
    for l in range(len(table) - 2, -1, -1):
        for t in range(len(table[l])):
            # the first term in the minimum is the time which may be allocated to the process according to the deadline and
            # current time, and the second term is the maximal accumulated time that the process may get and find a solution by
            # (the addition of the 1 is needed for the range to include this value)
            max_to_allocate = min(len(table[l]) - t, processes[dls_and_indices[l][1]].m.dist_list[-2][1] + 1)
            for j in range(max_to_allocate):
            # for j in range(len(table[l]) - t):
                val = table[l + 1][t + j][0] - LPF_i(processes[dls_and_indices[l][1]], j, 0)
                # if table[l][t] is None or table[l][t][0] < val:
                if table[l][t][0] < val:
                    table[l][t] = (val, j)
    # backtracking - build a schedule from the table
    schedule = []
    allocated_to_current = table[0][0][1] # table[0][0] is like OPT(0, 1) of the paper
    total_allocated = allocated_to_current
    for l in range(n):
        index = dls_and_indices[l][1]
        for _ in range(allocated_to_current):
            schedule.append(index)
        allocated_to_current = table[l + 1][total_allocated][1]
        total_allocated += allocated_to_current
    return schedule

class SAE2DynamicProgramming(Heuristic):
    '''This class represents the optimal heuristic gotten by a dynamic programming of an S(AE)^2 instance.'''

    def __init__(self, processes):
        # find the actual deadlines and get the order of allocations accordingly
        dls_and_indices = []
        for i, p in enumerate(processes):
            max_to_alloc = p.d.dist_list[-1][1]
            exec_blas_times = copmute_execution_blas_times_list(p, p.d.dist_list[-1][1])
            if len(exec_blas_times) > 0: # check if need to precede the deadline due to a base-level actions deadline
                max_to_alloc = max(0, min(max_to_alloc, exec_blas_times[0]))
            # if p.m.dist_list[-2][1] < max_to_alloc: # it's not helpful to allocate time if a process can't terminate anymore
            #     max_to_alloc = p.m.dist_list[-2][1]
            dls_and_indices.append((max_to_alloc, i))
        dls_and_indices.sort()
        self.schedule = dynamic_programming_sae2(processes, dls_and_indices)
        self.index = 0
    
    def get_next_action(self, state):
        if self.index < len(self.schedule):
            self.index += 1
            action = self.schedule[self.index - 1]
            while True:
                if state[1][action] != -math.inf:
                    return action
                self.index += 1
                if self.index < len(self.schedule):
                    action = self.schedule[self.index - 1]
                else:
                    break
        self.index = 0 # reset the index for reuse (it is needed when evaluting the heuristic and want using it again)
        return None # causes an overall failure


def compute_minimal_relevant_deadline(s_T, proc, sum_durs):
    cur_d = math.inf
    for (_, d) in proc.d.dist_list:
        if s_T + sum_durs <= d:
            cur_d = d
            break
    return cur_d

class RandomHeuristic(Heuristic):
    '''
    This class represents a heuristic which chooses a legal action randomly and uniformly and execute it.
    A computation action is legal if it doesn't invalidate the process, and a base-level action is legal if it is compatible
    with at least one process.
    '''

    def __init__(self, processes, base_level_actions):
        self.processes = processes
        self.base_level_actions = base_level_actions
        self.maximal_deadlines = [p.D.dist_list[-1][1] for p in processes]

    def get_next_action(self, state):
        can_be_allocated = []
        optional_blas = []
        for i in range(len(state[1])):
            if state[1][i] != -math.inf:
                if self.maximal_deadlines[i] > state[0]:
                    can_be_allocated.append(i)
                if len(self.processes[i].H) > state[3]:
                    optional_blas.append(self.processes[i].H[state[3]])
        if can_be_allocated == [] and optional_blas == []:
            return None
        action_index = random.randint(0, len(can_be_allocated) + len(optional_blas) - 1)
        if action_index < len(can_be_allocated):
            return can_be_allocated[action_index]
        return optional_blas[action_index - len(can_be_allocated)]


class MostPromisingPlanHeuristic(Heuristic):
    '''
    This class represents the greedy heuristic which chooses the currently most promising process and allocates it time until it
    fails. The heuristic is destinated for S(AE)^2 instances, since it doesn't execute base-level actions.
    '''

    def __init__(self, processes):
        self.current_process = None
        self.processes = processes
        self.processes_sum_durs = []
        for proc in processes:
            self.processes_sum_durs.append(compute_sum_of_durations(0, proc))
        self.maximal_deadlines = [p.D.dist_list[-1][1] for p in processes]
        self.current_minimal_relevant_deadline = 0
        self.current_to_allocate = 0
        self.current_allocated = 0

    def get_next_action(self, state):
        # if no process is currently chosen or the previous process has just failed, choose a process
        if self.current_process is None or state[1][self.current_process] == -math.inf:
            # choose the next process according to max_{p_i \in self.processes} sum_{(p,d) \in d_i} p * M_{p_i}(d)
            next_process = None
            next_process_prob = 0.0
            for i, proc in enumerate(self.processes):
                if state[1][i] != 0: # needed a process that hasn't failed and wasn't allocated computation time yet
                    continue
                # find the earliest relevant deadline
                cur_d = compute_minimal_relevant_deadline(state[0], proc, self.processes_sum_durs[i])
                prob = 0.0
                for (p, d) in proc.d.dist_list:
                    if d >= cur_d:
                        prob += p * proc.M.compute_probability(d)
                if next_process_prob < prob:
                    next_process = i
                    next_process_prob = prob
                    self.current_minimal_relevant_deadline = cur_d
            if next_process is None: # happens when all processes can't find a timely solution
                return None
            # last allocation is in the maximal time that the process can terminate and complete execute it's partial plan until
            # the maximal optional deadline
            for (_, m) in self.processes[next_process].m.dist_list:
                if m <= self.maximal_deadlines[next_process] - state[0] - self.processes_sum_durs[next_process]:
                    self.current_to_allocate = m
                else:
                    break
            self.current_process = next_process
        p_i = self.processes[self.current_process]
        if self.current_allocated == self.current_to_allocate or \
                state[0] + self.processes_sum_durs[self.current_process] >= self.maximal_deadlines[self.current_process]:
            state[1][self.current_process] = -math.inf # this is a kind of hack, since this function seemingly shouldn't change the
                                                       # state itself, but this heuristic will never allocate this process computation
                                                       # time, therefore it doesn't really make any difference
            self.current_process = None
            self.current_to_allocate = 0
            self.current_allocated = 0
            return self.get_next_action(state)
        self.current_allocated += 1
        return self.current_process


class ExecutiveMostPromisingPlanHeuristic(Heuristic):
    '''
    This class represents the greedy heuristic which chooses the currently most promising process and allocates it time until it
    fails. The base-level actions are executed lazily with respect to the earliest relevant deadline or to the deadline of the base-
    level actions.
    In more detail, the optional deadlines (times with a positive probability to be the deadlines) are those that enable the partial
    plan to be executed. This issue also has an effect on the success probability (rate of being promising) of the processes. Except
    for this, the deadlines of the base-level actions also must be met, so all the base-level actions are executed in the latest
    possible time in which the current and all the following actions are still applicable.
    '''

    def __init__(self, processes):
        self.current_process = None
        self.processes = processes
        self.maximal_deadlines = [p.D.dist_list[-1][1] for p in processes]
        self.current_minimal_relevant_deadline = 0
        self.current_to_allocate = 0
        self.current_allocated = 0
        self.execution_blas_times_list = []
    
    def get_next_action(self, state):
        # if no process is currently chosen or the previous process has just failed, choose a process
        if self.current_process is None or state[1][self.current_process] == -math.inf:
            # choose the next process according to max_{p_i \in self.processes} sum_{(p,d) \in d_i} p * M_{p_i}(d)
            next_process = None
            next_process_prob = 0.0
            for i, proc in enumerate(self.processes):
                if state[1][i] != 0: # needed a process that hasn't failed and wasn't allocated computation time yet
                    continue
                # find the earliest relevant deadline
                sum_durs = compute_sum_of_durations(state[3], proc)
                cur_d = compute_minimal_relevant_deadline(state[0], proc, sum_durs)
                prob = 0.0
                for (p, d) in proc.d.dist_list:
                    if d >= cur_d:
                        prob += p * proc.M.compute_probability(d)
                if next_process_prob < prob:
                    next_process = i
                    next_process_prob = prob
                    self.current_minimal_relevant_deadline = cur_d
            if next_process is None: # happens when all processes can't find a timely solution
                return None
            # last allocation is in the maximal time that the process can terminate until the maximal optional deadline
            for (_, m) in self.processes[next_process].m.dist_list:
                if m <= self.maximal_deadlines[next_process] - state[0]:
                    self.current_to_allocate = m
                else:
                    break
            self.current_process = next_process
            self.execution_blas_times_list = copmute_execution_blas_times_list(self.processes[self.current_process],
                                                                               self.current_minimal_relevant_deadline)
            self.execution_blas_times_list = self.execution_blas_times_list[state[3]:] # cut the times of former actions
        # check whether a base-level action should be ecexuted now
        p_i = self.processes[self.current_process]
        if len(self.execution_blas_times_list) > 0 and state[0] == self.execution_blas_times_list[0]:
            del self.execution_blas_times_list[0]
            return p_i.H[state[3]]
        sum_durs = compute_sum_of_durations(state[3], p_i)
        if self.current_allocated == self.current_to_allocate or state[0] + sum_durs >= self.maximal_deadlines[self.current_process]:
            state[1][self.current_process] = -math.inf # this is a kind of hack, since this function seemingly shouldn't change the
                                                       # state itself, but this heuristic will never allocate this process computation
                                                       # time, therefore it doesn't really make any difference
            self.current_process = None
            self.current_to_allocate = 0
            self.current_allocated = 0
            self.execution_blas_times_list = []
            return self.get_next_action(state)
        self.current_allocated += 1
        return self.current_process


class RoundRobinHeuristic(Heuristic):
    '''
    This class represents the round-robin heuristic for the S(AE)^2 problem: any of the processes gets a time unit for
    computation in turns.
    '''

    def __init__(self, processes):
        self.turn = 0
        self.n = len(processes)
        self.deadlines = [p.D.dist_list[-1][1] for p in processes]
        self.n_minus_infs = [-math.inf] * self.n
        self.processes = processes
    
    def get_next_action(self, state):
        if state[1] == self.n_minus_infs:
            return None
        for _ in range(self.n): # if no action is returned after n iterations another pass will be unhelpful
            if state[1][self.turn] == -math.inf: # process is failed
                self.turn = (self.turn + 1) % self.n
            else: # process is active
                p_i = self.processes[self.turn]
                sum_durs = compute_sum_of_durations(state[3], p_i)
                if state[0] + sum_durs + state[2] > self.deadlines[self.turn]: # process is irrelevant
                    state[1][self.turn] = -math.inf # the process will never become relevant back so the change doesn't harm
                                                    # anything but saves time for computing which action to return
                    self.turn = (self.turn + 1) % self.n
                    continue
                else:
                    break
        action = self.turn
        self.turn = (self.turn + 1) % self.n
        return action


class ExecutiveRoundRobinHeuristic(Heuristic):
    '''
    This class represents the round-robin heuristic for the IPAE problem: any of the processes gets a time unit for computation,
    and if a process should execute a base-level action to stay relevant, the base-level action is executed now, and in the next
    call the compution action will be returned and executed.
    '''

    def __init__(self, processes):
        self.turn = 0
        self.n = len(processes)
        self.deadlines = [p.D.dist_list[-1][1] for p in processes]
        self.n_minus_infs = [-math.inf] * self.n
        self.processes = processes
    
    def get_next_action(self, state):
        if state[1] == self.n_minus_infs:
            return None
        for _ in range(self.n): # if no action is returned after n iterations another pass will be unhelpful
            if state[1][self.turn] == -math.inf: # process is failed
                self.turn = (self.turn + 1) % self.n
            else: # process is active
                p_i = self.processes[self.turn]
                sum_durs = compute_sum_of_durations(state[3], p_i)
                if state[0] + sum_durs + state[2] > self.deadlines[self.turn]: # process is irrelevant
                    state[1][self.turn] = -math.inf # the process will never become relevant back so the change doesn't harm anything
                                                    # but saves time for computing which action to return
                    self.turn = (self.turn + 1) % self.n
                    continue
                # check whether a base-level action should be executed now
                minimal_relevant_deadline = compute_minimal_relevant_deadline(state[0], p_i, sum_durs)
                execution_blas_times_list = copmute_execution_blas_times_list(p_i, minimal_relevant_deadline)
                if len(execution_blas_times_list) > state[3] and execution_blas_times_list[state[3]] == state[0]:
                    if state[2] > 0:
                        state[1][self.turn] = -math.inf
                        self.turn = (self.turn + 1) % self.n
                    else:
                        return p_i.H[state[3]]
                else:
                    break
        action = self.turn
        self.turn = (self.turn + 1) % self.n
        return action


class BasicGreedySchemeHeuristic(Heuristic):
    '''
    This class represents the greedy scheme from the first paper for the S(AE)^2 problem.
    Note that the inputs for this heuristic are just inputs for IPAE without base-level actions.
    The implementation here follows the explanations from the second paper.
    '''

    def __init__(self, processes, alpha):
        self.processes = processes
        self.alpha = alpha
        self.cur_proc = None
        self.left_for_cur_proc = 0

    def Q_i(self, p, t_d, T_i):
        max_value = -math.inf
        best_t = None
        # the range of optional times is the minimum of time left until the deadline and the time from 1 to the last optional
        # termination due to time allocated
        for t in range(1, min(p.d.dist_list[-1][1] - t_d, p.m.dist_list[-2][1] - T_i) + 1):
            cur_value = self.alpha / p.d.E() - LPF_i(p, t, T_i, t_d) / t
            if cur_value > max_value:
                max_value = cur_value
                best_t = t
        return max_value, best_t

    def get_next_action(self, state):
        if self.left_for_cur_proc == 0 or self.cur_proc is None or state[1][self.cur_proc] == -math.inf:
            self.cur_proc = None # prevent allocating automatically another time for the process that was allocated previously
            max_Q = -math.inf
            for i, p in enumerate(self.processes):
                if state[1][i] == -math.inf:
                    continue
                cur_Q, cur_t = self.Q_i(p, state[0], state[1][i])
                if cur_Q > max_Q:
                    max_Q = cur_Q
                    self.left_for_cur_proc = cur_t
                    self.cur_proc = i
        self.left_for_cur_proc -= 1
        return self.cur_proc


def does_bla_needed_now(s, p_i):
    # check whether a base-level action should be executed now
    if s[2] == 0: # just to prevent heavy unneeded computations, the condition itself is unnecessary
        sum_durs = compute_sum_of_durations(s[3], p_i)
        minimal_relevant_deadline = compute_minimal_relevant_deadline(s[0], p_i, sum_durs)
        execution_blas_times_list = copmute_execution_blas_times_list(p_i, minimal_relevant_deadline)
        if len(execution_blas_times_list) > s[3] and execution_blas_times_list[s[3]] == s[0]:
            return True
    return False

class ExecutiveBasicGreedySchemeHeuristic(BasicGreedySchemeHeuristic):
    '''
    This class represents a version of the greedy scheme from the first paper fit to the IPAE problem.
    The fitting is made by executing base-level actions lazily during the allocation whenever it is needed.
    '''

    def __init__(self, processes, alpha):
        super().__init__(processes, alpha)

    def get_next_action(self, state):
        if self.left_for_cur_proc == 0 or state[1][self.cur_proc] == -math.inf:
            self.cur_proc = None # prevent allocating automatically another time for the process that was allocated previously
            max_Q = -math.inf
            for i, p in enumerate(self.processes):
                if state[1][i] == -math.inf:
                    continue
                cur_Q, cur_t = self.Q_i(p, state[0], state[1][i])
                if cur_Q > max_Q:
                    max_Q = cur_Q
                    self.left_for_cur_proc = cur_t
                    self.cur_proc = i
            if self.cur_proc is None:
                return None
        p_i = self.processes[self.cur_proc]
        if does_bla_needed_now(state, p_i):
            return p_i.H[state[3]]
        self.left_for_cur_proc -= 1
        return self.cur_proc


class DDAHeuristic(Heuristic):
    '''
    This class represents the DDA scheme for the S(AE)^2 problem.
    Note that the inputs for this heuristic are just inputs for IPAE without base-level actions.
    '''

    def __init__(self, processes, gamma, t_u):
        self.processes = processes
        self.gamma = gamma
        self.cur_proc = None
        self.left_for_cur_proc = 0
        self.t_u = t_u
    
    def Q_i(self, p, t_d, T_i):
        max_value = -math.inf
        # the range of optional times is the minimum of time left until the deadline and the time from 1 to the last optional
        # termination due to time allocated
        for t in range(1, min(p.d.dist_list[-1][1] - t_d, p.m.dist_list[-2][1] - T_i) + 1):
            cur_value = self.gamma * LPF_i(p, t, T_i, self.t_u + t) / (self.t_u + t) - LPF_i(p, t, T_i, t_d) / t
            if cur_value > max_value:
                max_value = cur_value
        return max_value

    def get_next_action(self, state):
        if self.left_for_cur_proc == 0 or state[1][self.cur_proc] == -math.inf:
            self.cur_proc = None # prevent allocating automatically another time for the process that was allocated previously
            max_Q = -math.inf
            for i, p in enumerate(self.processes):
                if state[1][i] == -math.inf:
                    continue
                cur_Q = self.Q_i(p, state[0], state[1][i])
                if cur_Q > max_Q:
                    max_Q = cur_Q
                    self.cur_proc = i
            self.left_for_cur_proc = self.t_u
        self.left_for_cur_proc -= 1
        return self.cur_proc


class DDAHeuristic2(Heuristic):
    '''
    This class represents the DDA scheme for the S(AE)^2 problem.
    Note that the inputs for this heuristic are just inputs for IPAE without base-level actions.
    '''

    def __init__(self, processes, gamma, t_u):
        self.processes = processes
        self.gamma = gamma
        self.cur_proc = None
        self.left_for_cur_proc = 0
        self.t_u = t_u

    def e_i(self, p, t_d, T_i):
        min_t = min(p.d.dist_list[-1][1] - t_d, p.m.dist_list[-2][1] - T_i) + 1
        min_val = math.inf
        for t in range(1, min_t):
            cur_val = LPF_i(p, t, T_i, t_d) / t
            if cur_val < min_val:
                min_t = t
                min_val = cur_val
        return min_t

    def Q_i(self, p, t_d, T_i):
        e_i_t_u = self.e_i(p, t_d + self.t_u, T_i)
        e_i_0 = self.e_i(p, t_d, T_i)
        first_element = 0 # e_i_t_u = 0 is equivalent to say that delaying the allocation will make p irrelevant
        if e_i_t_u != 0:
            first_element = self.gamma * LPF_i(p, e_i_t_u, T_i, t_d + self.t_u) / e_i_t_u
        second_element = 0 # e_i_0 = 0 is equivalent to say that p is already irrelevant
        if e_i_0 != 0:
            second_element = LPF_i(p, e_i_0, T_i, t_d) / e_i_0
        return first_element - second_element

    def get_next_action(self, state):
        if self.left_for_cur_proc == 0 or state[1][self.cur_proc] == -math.inf:
            self.cur_proc = None  # prevent allocating automatically another time for the process that was allocated previously
            max_Q = -math.inf
            for i, p in enumerate(self.processes):
                if state[1][i] == -math.inf:
                    continue
                cur_Q = self.Q_i(p, state[0], state[1][i])
                if cur_Q > max_Q:
                    max_Q = cur_Q
                    self.cur_proc = i
            self.left_for_cur_proc = self.t_u
        self.left_for_cur_proc -= 1
        return self.cur_proc


class DDAHeuristic3(Heuristic):
    '''
    This class represents the DDA scheme for the S(AE)^2 problem.
    Note that the inputs for this heuristic are just inputs for IPAE without base-level actions.
    '''

    def __init__(self, processes, gamma, t_u):
        self.processes = absorb_blas_to_deadlines(processes, None, [])
        self.gamma = gamma
        self.cur_proc = None
        self.left_for_cur_proc = 0
        self.t_u = t_u

    def Q_i(self, p, t_d, T_i):
        max_value = -math.inf
        # the range of optional times is the minimum of time left until the deadline and the time from 1 to the last optional
        # termination due to time allocated
        for t in range(1, min(p.d.dist_list[-1][1] - t_d, p.m.dist_list[-2][1] - T_i) + 1):
            cur_value = self.gamma * LPF_i(p, t, T_i, self.t_u + t) / (self.t_u + t) - LPF_i(p, t, T_i, t_d) / t
            if cur_value > max_value:
                max_value = cur_value
        return max_value

    def get_next_action(self, state):
        if self.left_for_cur_proc == 0 or state[1][self.cur_proc] == -math.inf:
            self.cur_proc = None  # prevent allocating automatically another time for the process that was allocated previously
            max_Q = -math.inf
            for i, p in enumerate(self.processes):
                if state[1][i] == -math.inf:
                    continue
                cur_Q = self.Q_i(p, state[0], state[1][i])
                if cur_Q > max_Q:
                    max_Q = cur_Q
                    self.cur_proc = i
            self.left_for_cur_proc = self.t_u
        self.left_for_cur_proc -= 1
        return self.cur_proc


class DDAHeuristic4(Heuristic):
    '''
    This class represents the DDA scheme for the S(AE)^2 problem.
    Note that the inputs for this heuristic are just inputs for IPAE without base-level actions.
    '''

    def __init__(self, processes, gamma, t_u):
        self.processes = processes
        self.gamma = gamma
        self.cur_proc = None
        self.left_for_cur_proc = 0
        self.t_u = t_u

    def e_i(self, p, t_d, T_i):
        min_t = min(p.d.dist_list[-1][1] - t_d, p.m.dist_list[-2][1] - T_i) + 1
        min_val = math.inf
        for t in range(1, min_t):
            cur_val = LPF_i(p, t, T_i, t_d) / t
            if cur_val < min_val:
                min_t = t
                min_val = cur_val
        return min_t

    def Q_i(self, p, t_d, T_i):
        e_i_t_u = self.e_i(p, t_d + self.t_u, T_i)
        e_i_0 = self.e_i(p, t_d, T_i)
        first_element = 0  # e_i_t_u = 0 is equivalent to say that delaying the allocation will make p irrelevant
        if e_i_t_u != 0:
            first_element = self.gamma * LPF_i(p, e_i_t_u, T_i, t_d + self.t_u) / e_i_t_u
        second_element = 0  # e_i_0 = 0 is equivalent to say that p is already irrelevant
        if e_i_0 != 0:
            second_element = LPF_i(p, e_i_0, T_i, t_d) / e_i_0
        return first_element - second_element

    def get_next_action(self, state):
        if self.left_for_cur_proc == 0 or state[1][self.cur_proc] == -math.inf:
            self.cur_proc = None  # prevent allocating automatically another time for the process that was allocated previously
            max_Q = -math.inf
            for i, p in enumerate(self.processes):
                if state[1][i] == -math.inf:
                    continue
                cur_Q = self.Q_i(p, state[0], state[1][i])
                if cur_Q > max_Q:
                    max_Q = cur_Q
                    self.cur_proc = i
            self.left_for_cur_proc = self.t_u
        self.left_for_cur_proc -= 1
        return self.cur_proc


class ExecutiveDDAHeuristic(DDAHeuristic):
    '''
    This class represents a version of DDA fit to the IPAE problem.
    The fitting is made by executing base-level actions lazily during the allocation whenever it is needed.
    '''

    def __init__(self, processes, gamma, t_u):
        super().__init__(processes, gamma, t_u)
    
    def get_next_action(self, state):
        if self.left_for_cur_proc == 0 or state[1][self.cur_proc] == -math.inf:
            self.cur_proc = None # prevent allocating automatically another time for the process that was allocated previously
            max_Q = -math.inf
            for i, p in enumerate(self.processes):
                if state[1][i] == -math.inf:
                    continue
                cur_Q = self.Q_i(p, state[0], state[1][i])
                if cur_Q > max_Q:
                    max_Q = cur_Q
                    self.cur_proc = i
            self.left_for_cur_proc = self.t_u
            if self.cur_proc is None:
                return None
        p_i = self.processes[self.cur_proc]
        if does_bla_needed_now(state, p_i):
            return p_i.H[state[3]]
        self.left_for_cur_proc -= 1
        return self.cur_proc


def compute_compatibility_groups(state, processes, num_of_blas):
    '''
    compute the group which compatible with a base-level action for any base-level action and the group of processes which allow
    delaying the execution of the next base-level action in a specific state (output size is |B|+1)
    '''
    groups = [[] for _ in range((num_of_blas + 1))]
    for i, p_i in enumerate(processes):
        if state[1][i] == -math.inf: # failed processes are not relevant
            continue
        if state[3] < len(p_i.H): # not all the actions in the partial plan were executed
            next_bla = p_i.H[state[3]]
            groups[next_bla.serial_number].append(i)
            # check whether the next action should not be executed now
            if state[2] != 0: # this condition just prevents heavy unnecessary computations many times
                groups[-1].append(i)
            else:
                sum_durs = compute_sum_of_durations(state[3], p_i)
                minimal_deadline_of_p = compute_minimal_relevant_deadline(state[0], p_i, sum_durs)
                maximal_time_of_next_bla = copmute_execution_blas_times_list(p_i, minimal_deadline_of_p)[state[3]]
                if state[0] != maximal_time_of_next_bla:
                    groups[-1].append(i)
        # all the actions in the partial plan were executed then just check that the maximal deadline won't be violted now
        elif state[0] < p_i.d.dist_list[-1][1]:
            groups[-1].append(i)
    return groups


# def __convert_groups_dist_to_list(groups_list, g_dict):
#     for key in g_dict.keys():
#         if key == None:
#             continue # the group of the current dictionary is added in the end to make sure that the group of empty series of base-
#                      # level actions will be the last in groups_list
#         else:
#             __convert_groups_dist_to_list(groups_list, g_dict[key])
#     if None in g_dict.keys():
#         groups_list.append(g_dict[None])
#     return groups_list

# def compute_forward_compatibility_groups(state, processes, forward_time):
#     '''
#     compute the compatibility groups of common prefix of actions that should be executed cosidering looking forward an amount of time
#     which is given as a parameter: a group of series of base-level actions B_g is declared when there is a process that must execute
#     this series of actions in the given forward time, and the group contains any process which B_g is a prefix of it's partial plan
#     when this process must execute just a B_g or a sub-series of B_g (note that any process might be in a few different groups and
#     the number of groups is bounded by |B|+1)
#     '''
#     groups_dict = {}
#     blas_to_exec = []
#     first_groups = [] # keeps the group with minimal series of base-level actions that is compatible with any of the processes
#     for p_i in processes:
#         # find the base-level actions that must be executed in the forward given time
#         sum_durs = compute_sum_of_durations(state[3], p_i)
#         minimal_relevant_deadline = compute_minimal_relevant_deadline(state[0], p_i, sum_durs)
#         execution_blas_times_list = copmute_execution_blas_times_list(p_i, minimal_relevant_deadline)
#         blas_to_exec.append([])
#         for i, exec_time in enumerate(execution_blas_times_list[state[3]:]):
#             if exec_time <= state[0] + forward_time:
#                 blas_to_exec[i].append(p_i.H[i])
#             else:
#                 break
#         # find the group of blas_to_exec[i] and add p_i to this group
#         g = groups_dict
#         for b in blas_to_exec[i]:
#             if not str(b) in g.keys():
#                 g[str(b)] = {}
#             g = g[str(b)]
#         if not None in g.keys():
#             g[None] = [p_i]
#         else:
#             g[None].append(p_i)
#         first_groups.append(g)
#     # just now that all the groups are already exist, the processes may be added to additional groups except for the minimal, namely
#     # first_groups[i]
#     for i, p_i in enumerate(processes):
#         g = first_groups[i]
#         # add p_i to any group that is also compatible with the continuance of the partial plan of p_i
#         for b in p_i.H[state[3] + len(blas_to_exec[i]):]:
#             if str(b) in g.keys():
#                 g = g[str(b)]
#                 if None in g.keys():
#                     g[None].append(p_i)
#             else:
#                 break
#     groups_list = __convert_groups_dist_to_list([], groups_dict)
#     return groups_list


class DDAOnMaxSizeHeuristic(DDAHeuristic):
    '''
    This class represents a composition of DDA (of S(AE)^2) on the decision of executing base-level actions according to the group of
    maximal size in the compatibility groups. 
    '''
    def __init__(self, processes, base_level_actions, gamma, t_u):
        super().__init__(processes, gamma, t_u)
        self.base_level_actions = base_level_actions
    
    def __choose_base_level_action_or_none(self, state):
        '''
        returns the base-level action of the maximal compatibility group, where None is returned when the maximal group is the last
        '''
        if state[2] > 0:
            return None
        compatibility_groups = compute_compatibility_groups(state, self.processes, len(self.base_level_actions))
        index_of_maximal_group = 0 # if it was initialized to len(compatibility_groups) - 1 instead of 0, ties would be broken by no
                                   # execution (namely None would be returned), but when a base-level action b is returned, it is
                                   # known surely that no processes will fail for incompatibility in the next dur(b) time units, while
                                   # in the other case we have no guarantee (except that after the current computation action there
                                   # will be the same number of active processes)
        for i in range(1, len(compatibility_groups)):
            if len(compatibility_groups[i]) > len(compatibility_groups[index_of_maximal_group]):
                index_of_maximal_group = i
        if index_of_maximal_group < len(compatibility_groups) - 1: # the last group contains the numbers of processes that are
                                                                   # compatible with a computation action (no base-level action
                                                                   # exection now)
            return self.base_level_actions[index_of_maximal_group]
        return None

    def get_next_action(self, state):
        next_bla = self.__choose_base_level_action_or_none(state)
        if next_bla is not None:
            return next_bla
        return super().get_next_action(state)


def absorb_blas_to_deadlines(processes, process_index, execution_times, s_T=0, s_L=0):
    '''
    given IPAE instance and series of base-level actions with their execution times, compute the equivalent instance of S(AE)^2
    '''
    sae2_instance = deepcopy(processes) # S(AE)^2 does not enable execution of base-level actions
    if process_index is not None:
        blas = sae2_instance[process_index].H
    else:
        blas = []
    for p_i in sae2_instance:
        # compute the invalidation time of p_i, and if needed, update p_i.d and p_i.D accordingly
        if len(p_i.H) == 0:
            if len(blas) == 0:
                continue
            else:
                invalidation_time = execution_times[0]
        else:
            sum_durs = compute_sum_of_durations(s_L, p_i)
            min_d = compute_minimal_relevant_deadline(s_T, p_i, sum_durs)
            p_i_exec_times = copmute_execution_blas_times_list(p_i, min_d)
            if len(blas) == 0:
                invalidation_time = p_i_exec_times[0]
            else:
                # p_i is invalidated when an incompatible base-level action is executed, but if blas is a prefix of p_i.H, p_i is
                # invalidated when it's next action should be executed, unless len(p_i.H) == len(blas), in which p_i is not
                # invalidated
                invalidation_time = min(execution_times[0], p_i_exec_times[0])
                compatible = True
                for i, b in enumerate(blas):
                    if i == len(p_i.H) or b != p_i.H[i]:
                        compatible = False
                        break
                    invalidation_time += b.duration
                if compatible and len(blas) == len(p_i.H):
                    p_i.H = [] # not changing p_i.d and p_i.D reflects that p_i.H is executed (on time)
                    continue
        # update p_i.d and p_i.D
        dist_list = []
        cummulative_prob = 0.0
        for (p, t) in p_i.d.dist_list:
            if t < invalidation_time:
                dist_list.append((p, t))
                cummulative_prob += p
            dist_list.append((1.0 - cummulative_prob, invalidation_time))
            break
        p_i.d = Distribution(dist_list)
        p_i.D = p_i.d.PMF_to_CDF()
        p_i.H = [] # changing p_i.d and p_i.D reflects the execution of the compatible prefix of p_i.H
    return sae2_instance

def compute_sae2_with_optimistic_deadlines(processes, process_index, s_T=0, s_L=0):
    '''
    make a reduction from IPAE to S(AE)^2 by setting the execution times of the base-level actions of the partial plan of the chosen
    process in the latest possible times as if the only processes exist are the input process and the current focused process
    '''
    p = processes[process_index]
    sum_durs = compute_sum_of_durations(s_L, p)
    min_d = compute_minimal_relevant_deadline(s_T, p, sum_durs)
    execution_times = copmute_execution_blas_times_list(p, min_d)
    return absorb_blas_to_deadlines(processes, process_index, execution_times)

class SandwichHeuristic(Heuristic):
    '''
    This class represents a heuristic that computes a schedule by bounding the probability of success in the case of known deadlines.
    '''

    def __init__(self):
        pass

    def get_next_action(self, state):
        pass


def isProcessCompatibleWithTimedBlas(p, timedBlas, state):
    '''check whether a process is compatible with a sequence of pairs of base-level actions and times to execute them'''
    if len(p.H) < len(timedBlas):
        return False
    sum_durs = compute_sum_of_durations(state[3], p)
    min_d = compute_minimal_relevant_deadline(state[0], p, sum_durs)
    execution_times = copmute_execution_blas_times_list(p, min_d)
    for i in range(len(timedBlas)):
        if execution_times[i] < timedBlas[i][1]:
            return False
    return True

class JustDoSomethingHeuristic(Heuristic):
    '''
    This class represents a heuristic that computes a schedule by setting base-level actions and times gradually greedily in the
    known deadlines case.
    The idea is to check what happens if u actions are executed according to the partial plan of any of the processes using
    absorption of the actions to the d_i distributions, and then to check the next u actions and so on. The check for which option
    is the best is based on a heuristic for S(AE)^2 that is given as a parameter to the constructor.
    '''
    
    def __init__(self, heuristic_name, processes, blas, u, additional_params=None):
        self.processes = processes
        self.blas = blas
        self.u = u
        self.additional_params = additional_params
        self.he_name = heuristic_name
        self.he = build_heuristic(self.he_name, self.processes, blas, self.additional_params)
        self.index = 0
        self.timedBlas = None
        self.chosen_process = None
        
    def get_next_action(self, state):
        if self.timedBlas is None: # compute schedule in the first call
            # check what happens if no base-level actions are executed (DDA evaluation)
            # the dummy process is needed to fail all the processes when they should start execution, and it cannot find a solution
            dummy_d = Distribution([(1.0,1)])
            dummy_m = Distribution([(0.5,2), (0.5,3)]) # cannot succeed since the deadline is at 1
            best_acc = [] # keeps the overall best option found
            dummy_process = Process(dummy_m, dummy_d, best_acc)
            self.processes.append(dummy_process)
            # keeps the probability of best_acc
            self.he.processes = self.processes
            self.he = build_heuristic(self.he_name, self.processes, self.blas, self.additional_params)
            best_prob = evaluate_heuristic_known_deadlines(self.processes, self.he)
            del self.processes[-1] # delete the dummy process
            # check iteratively whether it is profitable to execute u additional base-level actions
            compatible_processes = [p for p in self.processes]
            prev_iter_best_acc = [] # keeps the best option of the previous iteration
            while True:
                # find the process in which the next u actions are the most profitable
                local_best_acc = None       # keeps the best option of the last (while) iteration
                local_best_prob = -math.inf # keeps the prob of local_best_acc
                acc_copy = [a_t for a_t in prev_iter_best_acc] # used to add the u (or less) actions for any process
                for cur_proc, p in enumerate(compatible_processes):
                    if len(p.H) <= len(prev_iter_best_acc):
                        continue
                    # remove the actions added in the eprevious iteration and add the next actions with lazy execution times
                    acc_copy = acc_copy[:len(prev_iter_best_acc)]
                    # extend acc_copy by u actions or less than u if there are less then u actions to add
                    add_from = len(acc_copy)                       # including add_from
                    add_to = min(len(acc_copy) + self.u, len(p.H)) # not including add_to
                    sum_durs = compute_sum_of_durations(state[3], p)
                    min_d = compute_minimal_relevant_deadline(state[0], p, sum_durs)
                    execution_times = copmute_execution_blas_times_list(p, min_d)
                    action_time_additional_pairs = [(p.H[i], execution_times[i]) for i in range(add_from, add_to)]
                    acc_copy.extend(action_time_additional_pairs)
                    # reduce to S(AE)^2 and compute the probability of success
                    dummy_process.H = [b for b in p.H[:add_to]]
                    self.processes.append(dummy_process)
                    procs = absorb_blas_to_deadlines(self.processes, len(self.processes) - 1, execution_times, state[0], state[3])
                    self.he = build_heuristic(self.he_name, procs, self.blas, self.additional_params)
                    self.he.processes = procs
                    acc_copy_prob = evaluate_heuristic_known_deadlines(procs, self.he)
                    del self.processes[-1]
                    # check if this is a first local option or a better option was found (the >= is needed to make sure that there
                    # is a process with the base-level actions chosen, and the same is correct for the next if statement)
                    if acc_copy_prob >= local_best_prob:
                        local_best_acc = acc_copy
                        local_best_prob = acc_copy_prob
                        if acc_copy_prob >= best_prob: # check if better than current best global option
                            best_acc = acc_copy
                            best_prob = acc_copy_prob
                            self.chosen_process = cur_proc
                if local_best_acc is None: # no more options exist
                    break
                prev_iter_best_acc = local_best_acc
                tmp_compatible_processes = []
                for p in compatible_processes:
                    if isProcessCompatibleWithTimedBlas(p, prev_iter_best_acc, state):
                        tmp_compatible_processes.append(p)
                compatible_processes = tmp_compatible_processes
            self.timedBlas = best_acc
            self.execution_times = [t for (_, t) in self.timedBlas]
        # when having times of base-level actions execution, just check whether it's a time to execute an action, otherwise use DDA
        if self.index < len(self.timedBlas) and self.timedBlas[self.index][1] == state[0]:
            self.index += 1
            return self.timedBlas[self.index - 1][0]
        # reduce to S(AE)^2 to be able to use the DDA heuristic
        procs = absorb_blas_to_deadlines(self.processes, self.chosen_process, self.execution_times, state[0], state[3])
        self.he = build_heuristic(self.he_name, procs, self.blas, self.additional_params)
        return self.he.get_next_action(state)


def compress_processes(processes, k):
    '''
    return a copy of processes after compress the m_i distributions of processes to have no more than k termination times,
    excluding the time of probability 1
    '''
    compressed = deepcopy(processes)
    for p in compressed:
        len_p_M = len(p.M.dist_list)
        if len_p_M <= k + 1:
            continue
        dist = []
        indices = [int(len_p_M - 2 - i * (len_p_M - 1) / k) for i in range(k)]
        for i in indices:
            dist.insert(0, p.M.dist_list[i])
        dist.append(p.M.dist_list[-1])
        p.M = Distribution(dist, False)
        p.m = p.M.CDF_to_PMF()
    return compressed


class LazinessHeuristic(Heuristic):
    '''
    This class represents the heuristic in which any of the partial plans is checked with lazy execution times by a reduction
    to S(AE)^2 and a heuristic of S(AE)^2. The best lazy execution times are chosen and then we act accordingly with the
    heuristic that was used for the estimation.
    '''

    def __init__(self, processes, blas, heuristic_name, additional_params=None):
        self.processes = processes
        # check the option of not executing any base-level action
        best_sae2_instance = absorb_blas_to_deadlines(self.processes, None, [])
        self.best_pp = []
        self.execution_times = []
        # self.he = build_heuristic(heuristic_name, self.processes, blas, additional_params)
        self.he = build_heuristic(heuristic_name, best_sae2_instance, blas, additional_params)
        best_prob = evaluate_heuristic_known_deadlines(self.processes, self.he)
        for i, p in enumerate(processes):
            sum_durs = compute_sum_of_durations(0, p)
            min_d = compute_minimal_relevant_deadline(0, p, sum_durs)
            exec_times = copmute_execution_blas_times_list(p, min_d)
            p_sae2_instance = absorb_blas_to_deadlines(processes, i, exec_times)
            he = build_heuristic(heuristic_name, p_sae2_instance, blas, additional_params)
            p_prob = evaluate_heuristic_known_deadlines(p_sae2_instance, he)
            if p_prob > best_prob:
                best_prob = p_prob
                # self.best_sae2_instance = p_sae2_instance
                self.best_pp = p.H
                self.execution_times = exec_times
                self.he = he

    def get_next_action(self, state):
        if state[3] < len(self.execution_times) and state[0] == self.execution_times[state[3]]:
            return self.best_pp[state[3]]
        return self.he.get_next_action(state)


class MCTSHeuristic(Heuristic):
    '''
    This class represents a Monte Carlo tree search algorithm.
    Any node in the tree has the following structure: [state, blas, rollouts_number, accumulated_value, action, children], where:
    state has the regular form of (T, allocations, W, L); blas are the base-level actions that appear as the next action in the
    partial plan of some process; rollouts_number is the number of rollouts played from this node or offsprings of this node;
    accumulated_value is the sum of values gotten from the rollouts; action is the action that led to this node from parent node;
    children is the nodes that where created from the expansion of this node or [] if the node hasn't yet been expanded.
    '''

    def __init__(self, processes, num_of_rollouts, C=math.sqrt(2)):
        self.processes = processes
        self.n = len(processes)
        self.C = C # the exploration-exploitation tuning constant for the UCB1 calculation
        self.num_of_rollouts = num_of_rollouts
        first_blas = [] # blas which leave at least on validate process
        for p in self.processes:
            if len(p.H) > 0 and not p.H[0] in first_blas:
                first_blas.append(p.H[0])
        self.root = [(0, [0] * self.n, 0, 0), first_blas, 0, 0, None, []]
        self.former_actions = [] # used for computing the value parameter of the function mcts
        self.sampled_deadlines = None
        self.maximal_execution_times = []
        for p in self.processes:
            # sum_durs = compute_sum_of_durations(0, p)
            # min_d = compute_minimal_relevant_deadline(0, p, sum_durs)
            # self.execution_times.append(copmute_execution_blas_times_list(p, min_d))
            self.maximal_execution_times.append(copmute_execution_blas_times_list(p, p.d.dist_list[-1][1]))
        self.real_execution_times = None # keep the deadlines that are sampled for the current simulation
        # keep the situation of the processes according to the sampled deadlines: valid - true, invalid - false
        # the situation of any process may change after any action, therefore a loop follows any action taken
        self.real_valid = None

    def mcts(self):
        for _ in range(self.num_of_rollouts): # make the rollouts
            # sampling deadlines and calculating the lazy calculation times according to those sampled deadlines
            self.sampled_deadlines = [p.D.sample() for p in self.processes]
            self.real_execution_times = []
            for i, p in enumerate(self.processes):
                self.real_execution_times.append(copmute_execution_blas_times_list(p, self.sampled_deadlines[i]))
            self.real_valid = [True] * self.n
            # the rollout value should consider also the actions that were already taken from the initial state
            rollout_value = 0
            former_allocations = [0] * self.n
            former_blas_counter = 0
            for i, a in enumerate(self.former_actions):
                if isinstance(a, BaseLevelAction):
                    for j, p in enumerate(self.processes): # invalidating all incompatible processes
                        if self.real_valid[j] and (len(p.H) <= former_blas_counter or p.H[former_blas_counter] != a):
                            self.real_valid[j] = False
                    former_blas_counter += 1
                    continue
                # check whether the process is still valid according to the real deadlines, if it shouldn't be invalidated
                # beacuse of the base-level action that where already executed (note that by the following implementation)
                # we know that no tardy actions are allocated), and if the process is still valid according to it's sampled
                # deadline (needed for the case that all it's partial plan has been executed)
                if self.real_valid[a] and (former_blas_counter == len(self.real_execution_times[a]) or \
                        i - former_blas_counter < self.real_execution_times[a][former_blas_counter]) and \
                        i - former_blas_counter < self.sampled_deadlines[a]:
                    rollout_value -= LPF_i(self.processes[a], 1, former_allocations[a], i - former_blas_counter)
                for j, p in enumerate(self.processes): # invalidating any process that needed a base-level action execution now
                    if self.real_valid[j] and len(p.H) > former_blas_counter and \
                            i - former_blas_counter == self.real_execution_times[j][former_blas_counter]:
                        self.real_valid[j] = False
                former_allocations[a] += 1
            # initial the search for the simulation
            current_node = self.root
            ancestors = [current_node] # needed for backpropagating the value of the rollout in the end
            while current_node[-1] != []: # going down to a leaf using the UCB1 formula
                next_node = self.choose_best_child(current_node, current_node[2])
                action = next_node[4]
                # check if this is a computation action that is given to a valid process before the sampled deadline
                if isinstance(action, int) and self.real_valid[action] and \
                        current_node[0][0] < self.sampled_deadlines[action] and current_node[0][1][action] != -math.inf:
                    # check if the next base-level action in the partial plan, if exists, should haven't been already executed
                    if len(self.processes[action].H) == current_node[0][3] or \
                            (len(self.processes[action].H) > current_node[0][3] and \
                            current_node[0][0] < self.real_execution_times[action][current_node[0][3]]):
                        rollout_value -= LPF_i(self.processes[action], 1, current_node[0][1][action], current_node[0][0])
                # invalidate processes as described above
                if isinstance(action, int):
                    for i, p in enumerate(self.processes):
                        if self.real_valid[i] and len(p.H) > current_node[0][3] and \
                                current_node[0][0] == self.real_execution_times[i][current_node[0][3]]:
                            self.real_valid[i] = False
                else:
                    for i, p in enumerate(self.processes):
                        if self.real_valid[i] and (len(p.H) <= current_node[0][3] or p.H[current_node[0][3]] != action):
                            self.real_valid[i] = False
                current_node = next_node
                ancestors.append(current_node)
            if current_node[2] != 0: # checking whether the node hasn't been estimated yet and therefore should be expanded
                current_node[-1] = self.expand(current_node)
                # check whether the expansion hasn't led to a node with no valid processes, and if there's no child to make
                # the rollout from, the rollout will be made from this node
                if current_node[-1] != []:
                    # any of the children we have gotten from the exapnsion has a UCB1 value of infinity then take the first
                    action = current_node[-1][0][4]
                    # same nested conditions as in the while loop above
                    if isinstance(action, int) and self.real_valid[action] and \
                            current_node[0][0] < self.sampled_deadlines[action] and \
                            current_node[0][1][action] != -math.inf:
                        if len(self.processes[action].H) == current_node[0][3] or \
                                (len(self.processes[action].H) > current_node[0][3] and \
                                current_node[0][0] < self.real_execution_times[action][current_node[0][3]]):
                                # current_node[0][0] < self.maximal_execution_times[action][current_node[0][3]]):
                            rollout_value -= LPF_i(self.processes[action], 1, current_node[0][1][action], current_node[0][0])
                    # same invalidation according to the real deadlines as above
                    # note that the next condition will be always true, since the expansion considers the computation action
                    # first, and if there's a child that is a result of a base-level action, than there's a valid process,
                    # so this process may be allocated, so there's a child that is a result of this computation action, and it
                    #  appears before the child that is a result of the base-level action
                    if isinstance(action, int):
                        for i in range(self.n):
                            if current_node[0][3] < len(self.real_execution_times[i]) and \
                                    current_node[0][0] == self.real_execution_times[i][current_node[0][3]]:
                                self.real_valid[i] = False
                    else:
                        for i, p in enumerate(self.processes):
                            if self.real_valid[i] and (len(p.H) <= current_node[0][3] or p.H[current_node[0][3]] != action):
                                self.real_valid[i] = False
                    current_node = current_node[-1][0]
                    ancestors.append(current_node)
            # the rollout value is calculated in this function for the deterministic part and in the function follout for the
            # non-deterministic part
            rollout_value += self.rollout(current_node)
            for anc in ancestors: # backpropagating rollout_value to the nodes on the path from the root
                anc[2] += 1
                anc[3] += rollout_value

    def UCB1(self, node, parent_rollouts):
        if node[2] == 0:
            return math.inf
        return node[3] / node[2] + self.C * math.sqrt(math.log(parent_rollouts) / node[2])

    def choose_best_child(self, node, parent_rollouts):
        best_child_index = -1
        max_UCB1 = -1
        for i, child in enumerate(node[-1]):
            child_UCB1 = self.UCB1(child, parent_rollouts)
            if child_UCB1 > max_UCB1:
                best_child_index = i
                max_UCB1 = child_UCB1
        return node[-1][best_child_index]

    def expand(self, node):
        children = []
        for i in range(self.n):
            comp_children = self.create_node_by_action(node, i)
            for comp_child in comp_children:
                children.append(comp_child)
        for b in node[1]:
            children.append(self.create_node_by_action(node, b)[0])
        return children

    def create_node_by_action(self, node, a):
        if node[0][1] == [-math.inf] * self.n: # if all the processes are failing the are no valid children
            return []
        if isinstance(a, int):
            if node[0][1][a] == -math.inf: # don't consider tardy computation actions
                return []
            allocations = deepcopy(node[0][1])
            allocations[a] += 1
            # check invalidation of any of the processes (including a) due to deadline conditions after the execution of a
            # note that the deadlines used here are the maximal deadlines and not the (unknown) sampled deadlines
            for i, p in enumerate(self.processes):
                if (len(self.processes[a].H) > node[0][3] and self.maximal_execution_times[a][node[0][3]] == node[0][0]) or \
                    self.processes[i].d.dist_list[-1][1] == node[0][0] + 1:
                    allocations[i] = -math.inf
            # invalidate the process if got the maximal accumulated time by which it can terminate successfully and thus prevent
            # tardy allocations to this process in the future
            if allocations[a] == self.processes[a].m.dist_list[-2][1]:
                allocations[a] = -math.inf
            next_blas = []
            # check whether no base-level will be in execution after this computation action is finished, otherwise no base-
            # level action can be executed immediately after this computation action
            if node[0][2] <= 1:
                for i, p in enumerate(self.processes):
                    if allocations[i] != -math.inf and len(p.H) > node[0][3] and not p.H[node[0][3]] in next_blas:
                        next_blas.append(p.H[node[0][3]])
            next_state = (node[0][0] + 1, allocations, max(0, node[0][2] - 1), node[0][3])
            next_nodes = [[next_state, next_blas, 0, 0, a, []]]
            # either the allocated process stays valid after the allocation or not, and if it may stay valid but also may
            # fail, the returned list will contain both options, but if it failed we can stop here, because next we just
            # check whether it may (also) fail
            if next_state[1][a] == -math.inf:
                return next_nodes
            may_fail = False
            rest_dur = 0
            for b in self.processes[a].H[node[0][3]:]:
                rest_dur += b.duration
            # check if the deadline may be discovered to have been expired before finishing the current allocation
            if self.processes[a].d.dist_list[0][1] - rest_dur <= node[0][0]:
                # check if the process may terminate by the current allocation, otherwise we won't figure out the real deadline
                for (_, t) in self.processes[a].m.dist_list:
                    if allocations[a] == t:
                        may_fail = True
                        break
            if may_fail:
                allocations2 = deepcopy(allocations)
                allocations2[a] = -math.inf
                next_blas2 = []
                if node[0][2] <= 1:
                    for i, p in enumerate(self.processes):
                        if allocations2[i] != -math.inf and len(p.H) > node[0][3] and not p.H[node[0][3]] in next_blas2:
                            next_blas2.append(p.H[node[0][3]])
                next_state2 = (node[0][0] + 1, allocations2, max(0, node[0][2] - 1), node[0][3])
                next_nodes.append([next_state2, next_blas2, 0, 0, a, []])
            return next_nodes
        else: # a is a base-level action
            allocations = deepcopy(node[0][1])
            for i, p in enumerate(self.processes): # invalidating all incompatible processes
                if allocations[i] != -math.inf and (len(p.H) <= node[0][3] or p.H[node[0][3]] != a):
                    allocations[i] = -math.inf
            next_state = (node[0][0], allocations, a.duration, node[0][3] + 1)
            # note that at least one process stays valid, since the chosen base-level action must be such a one that appears in
            # a partial plan of at least one of the processes
            return [[next_state, [], 0, 0, a, []]]

    def rollout(self, node):
        rollout_value = 0
        while node != []: # check whether there are valid processes yet
            # list of valid processes, namely list of valid computation actions
            valid_procs_indices = [i for i in range(self.n) if node[0][1][i] != -math.inf]
            range_of_sample = len(valid_procs_indices) + len(node[1])
            if range_of_sample == 0: # no valid actions available
                return rollout_value
            index = random.randrange(range_of_sample) # sample an action from all vaild actions uniformly
            # a helper variable for the -LPF_i value is declared for the case that the next (computation) action will invalidate
            # all the processes, and therefore the -LPF_i value shouldn't be added and will be subtracted from the returned
            # value (i.e. it shows that the computation action couldn't help this process to find a solution, that is tardy)
            absolute_lpf_i_value = 0
            if index < len(valid_procs_indices): # a computation action was sampled
                action = valid_procs_indices[index]
                # check whether the process will stay valid after being allocated
                if self.real_valid[action] and node[0][0] < self.sampled_deadlines[action]:
                    # check whether no base-level action should have been executed for this process now
                    if len(self.processes[action].H) == node[0][3] or \
                            (len(self.processes[action].H) > node[0][3] and \
                            node[0][0] < self.real_execution_times[action][node[0][3]]):
                        absolute_lpf_i_value = -LPF_i(self.processes[action], 1, node[0][1][action], node[0][0])
                        rollout_value += absolute_lpf_i_value
                # update the validation situation according to the real deadline for the rollout value computations
                for i in range(self.n):
                    if node[0][3] < len(self.real_execution_times[i]) and node[0][0] == self.real_execution_times[i][node[0][3]]:
                        self.real_valid[i] = False
            else: # a base-level action was sampled
                action = node[1][index - len(valid_procs_indices)]
                # update the real validation situation
                for i, p in enumerate(self.processes):
                    if self.real_valid[i] and (len(p.H) <= node[0][3] or p.H[node[0][3]] != action):
                        self.real_valid[i] = False
            next_nodes = self.create_node_by_action(node, action)
            # the next node should be the only next node if there is only one option, otherwise it should be the node in which
            # the process has or hasn't failed due to the deadline according to the sampled deadline
            if len(next_nodes) == 1 or (len(next_nodes) == 2 and (isinstance(action, int) and \
                    node[0][0] + 1 <= self.sampled_deadlines[action])):
                node = next_nodes[0]
            elif len(next_nodes) == 2:
                node = next_nodes[1]
            else: # no child with valid processes exists, and the -LPF_i should be subtracted as described above
                return rollout_value - absolute_lpf_i_value
        return rollout_value

    def get_next_action(self, state):
        self.mcts() # make rollouts from the current root
        best_action = None # finding the best action to choose according to the average of the estimated values
        best_average = -1
        for child in self.root[-1]:
            if child[2] != 0: # any child that hasn't been estimated is not chosen
                child_average = child[3] / child[2]
                if child_average > best_average:
                    best_action = child[4]
                    best_average = child_average
        # we commited to an action, therefore we should consider just the paths that consider taking that action
        self.former_actions.append(best_action)
        for child in self.root[-1]:
            if child[4] == best_action:
                self.root = child
                break
        return best_action


class KBoundedHeuristic(Heuristic):
    '''
    This heuristic runs a given S(AE)^2 heuristic for any option of timing the head of size k of any of the processes. If there
    are more than k base-level action in a partial plan of some process, those excess actions are executed lazily.
    Given a base-level action b, the only times that should be considered for executing b are the lazy times that b should be
    executed at according to any of the processes that are compatible with the former base-level actions.
    The heuristic works optimally if lengths of all the partial plans are bounded by the given k, if the heuristic is the DP and
    if the deadlines are known.
    '''

    def __init__(self, processes, base_level_actions, k, heuristic_name, additional_params=None):
        lazy_times = [] # keeps the lazy execution times of the base-level actions in the partial plan of any of the processes
        for p in processes:
            # lazy_times.append(copmute_execution_blas_times_list(p, p.d.dist_list[-1][1]))
            lazy_times.append(copmute_execution_blas_times_list(p, p.d.dist_list[0][1]))
        # help to avoid multiple reductions when more than one process have empty partial plan
        evaluated_empty_partial_plan = False
        self.he = None
        best_prob = -1
        self.timing = None
        self.chosen_H = None
        for i in range(len(processes)):
            # a list of lists that keeps in the j'th list all the times that the j'th action in the partial plan of the chosen
            # process may be executed, but not after the lazy execution time according to the chosen process
            cur_k = min(k, len(processes[i].H))
            compatible_lazy_times = [[] for _ in range(cur_k)]
            p_H = processes[i].H
            for j1, p in enumerate(processes):
                for j2, b in enumerate(p.H[:cur_k]):
                    if b != p_H[j2]:
                        break
                    # take times just if they don't exceed from the lazy execution time for process i and haven't appeared yet
                    if lazy_times[i][j2] >= lazy_times[j1][j2] and not lazy_times[j1][j2] in compatible_lazy_times[j2]:
                        compatible_lazy_times[j2].append(lazy_times[j1][j2])
            # the cartesian product of the lists in compatible_lazy_times include all the legal combinations of the timings
            p_i_exec_times_combinations = [list(timing) for timing in itertools.product(*compatible_lazy_times)]
            # the combinations may include illegal timings for overlap in execution, so here we throw such timings, and we do
            # that from the end to the beginning (regarding to the indices), since the indices of the timings are changed when
            # erasing an illegal timing
            for j1 in range(len(p_i_exec_times_combinations) - 1, -1, -1):
                timing = p_i_exec_times_combinations[j1]
                for j2 in range(len(timing) - 1):
                    if timing[j2] + p_H[j2].duration > timing[j2 + 1]: # the check of overlap in execution
                        del p_i_exec_times_combinations[j1]
                        break
            # if the partial plan of the chosen process is empty, we would like to run the heuristic as if the input is sae2
            # instance, and we shouldn't do that more than once (would have happened if there are more than one process with
            # empty partial plan)
            if p_i_exec_times_combinations == [] and not evaluated_empty_partial_plan:
                evaluated_empty_partial_plan = True
                p_i_exec_times_combinations.append([])
            # finding the best timing option for the k-lengthed prefix of the partial plan of this process
            for cur_exec_times in p_i_exec_times_combinations:
                cur_exec_times.extend(lazy_times[i][cur_k:])
                cur_reduced_to_sae2 = absorb_blas_to_deadlines(processes, i, cur_exec_times)
                he = build_heuristic(heuristic_name, cur_reduced_to_sae2, base_level_actions, additional_params)
                cur_prob = evaluate_heuristic_known_deadlines(cur_reduced_to_sae2, he)
                if cur_prob > best_prob:
                    self.he = he
                    best_prob = cur_prob
                    self.timing = cur_exec_times
                    self.chosen_H = processes[i].H

    def get_next_action(self, state):
        if state[3] < len(self.timing) and state[0] == self.timing[state[3]]:
            return self.chosen_H[state[3]]
        return self.he.get_next_action(state)


def compute_optimal_probability_known_deadlines(processes, base_level_actions):
    max_H_len = 0
    for p in processes:
        if len(p.H) > max_H_len:
            max_H_len = len(p.H)
    runtime = time.time()
    kbnd_he = KBoundedHeuristic(processes, base_level_actions, max_H_len, "SAE2DynamicProgramming")
    runtime = time.time() - runtime
    schedule = kbnd_he.he.schedule # initially it contains just the computation actions
    for  i in range(len(kbnd_he.chosen_H)):# insert the blas to the schedule
        schedule.insert(kbnd_he.timing[i], kbnd_he.chosen_H[i])
    prob = evaluate_schedule_known_deadlines(processes, schedule)
    return prob, runtime


def compute_optimal_probability_known_deadlines_file_input(folder, input):
    processes, base_level_actions, _, _ = build_input_from_text_file(folder + "/" + input)
    return compute_optimal_probability_known_deadlines(processes, base_level_actions)


def from_ud_to_kd_deadlines(processes, conversion="expectation"):
    '''convert an unknown deadlines instance into a known deadlines instance according to a given type of conversion'''
    processes = deepcopy(processes)
    for p in processes:
        if conversion == "expectation":
            # the expectation may be non-integer number, and we would like to choose a number with a positive probability to
            # be the deadline, then if the part after the period is less then 5, the deadline is chosen to be the last number
            # with a positive probability to be the deadline before the expectation, otherwise the first number after
            p_d_E = p.d.E()
            dl = -1
            prev_t = -1
            for _, t in p.d.dist_list:
                if t >= p_d_E:
                    dl = t if p_d_E - math.floor(p_d_E) >= 0.5 else prev_t
                    break
                prev_t = t
            p.d = Distribution([(1, dl)])
        elif conversion == "minimum":
            p.d = Distribution([(1, p.d.dist_list[0][1])])
        elif conversion == "maximum":
            p.d = Distribution([(1, p.d.dist_list[-1][1])])
        p.D = p.d.PMF_to_CDF()
    return processes