import math
from copy import deepcopy
import numpy as np
import random
import re

# from mdp import MDP
from mdp import MDP2


'''
In this module defined the classes needed for an MDP for the IPAE problem where:
    1. The deadlines are uncertain;
    2. The durations of the super-actions are uncertain;
    3. Reversibility is not assumed;
    4. The durations of base-level actions are fixed known numbers.
In addition, it is assumed W.L.O.G. that P_i = 1 and that R_i = 0 for all i in {1...n}.
'''

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
            if st.T >= deadlines[a] or st.T_i[a] == -math.inf:
                return []
            st.T += 1
            st.T_i[a] += 1
            st.W = max(0, st.W-1)
            for i, p in enumerate(self.processes): # check whether the partial plan is still applicable for any of the processes
                earliest_time_to_done_exec = st.T + st.W
                for b in p.H[st.L:]:
                    earliest_time_to_done_exec += b.duration
                    if earliest_time_to_done_exec > b.deadline:
                        st.T_i[i] = -math.inf
                        break
            if st.T_i[a] == -math.inf: # next part is redundant if the process is failed
                return [st]
            sts = [st]
            # the process may fail only in case that the current time has a positive probability in the m distribution
            termination_times = [t for _, t in self.processes[a].m.dist_list]
            if st.T_i[a] in termination_times: # computation action may cause a completion of computation which might fail the process
                st_F = deepcopy(st)
                st_F.T_i[a] = -math.inf
                sts.append(st_F)
            return sts
        else: # a is a base-level action
            if st.W > 0 or st.T_i == n_minus_infs:
                return []
            st.W = a.duration
            st.L += 1
            for i in range(0,n):
                if st.L > len(self.processes[i].H) or not self.processes[i].H[st.L-1] == a:
                    st.T_i[i] = -math.inf
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
                    if ns.T_i[a] == -math.inf: # the process cannot meet the deadline of some action in it's partial plan
                        transitions[s][a] = [(1.0, self.__states_dict[hash(ns)])]
                    else:
                        p_i = self.processes[a]
                        sum_durs = 0
                        for l in range(ns.L, len(p_i.H)):
                            sum_durs += p_i.H[l].duration
                        q_i_ns = ns.T + ns.W + sum_durs - 1
                        p_Ci_s_ns = p_i.m.compute_probability(ns.T_i[a]) / (1 - p_i.M.compute_probability(s.T_i[a]))
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
            if st.T == math.inf: # the reward of SUCCESS is 1
                rewards[st] = 1.0
            else: # the reward of any other state is 0
                rewards[st] = 0.0
        return rewards

    def __evaluate_policy_rec(self, policy, state):
        if state == self.__SUCCESS:
            return 1.0
        if state == self.__FAIL:
            return 0.0
        action = policy[self.__states_dict[hash(state)]]
        prob = 0.0
        for (p, s) in self.transitions[state][action]:
            prob += p * self.__evaluate_policy_rec(policy, s)
        return prob

    def evaluate_policy(self, policy):
        '''
        Compute the probability of finding a timely solution by following a policy.
        '''
        return self.__evaluate_policy_rec(policy, self.__states_dict[hash(self.init)])


    class State:
        '''This class represents a state in the MDP.'''
        def __init__(self, absolute_time, allocated_times, waiting_time, leading_actions):
            self.T = absolute_time      # the absolute time that has passed if not a terminal state, infinity - SUCCESS, -infinity -
                                        # FAIL
            self.T_i = allocated_times  # list where the i'th cell represents the computation time allocated to process i, -infinity
                                        # if failed
            self.W = waiting_time       # 0 if not base-level action is currently executed, otherwise the time remained for the action
                                        # to be done
            self.L = leading_actions    # number of base-level actions that were taken, including the current action if such exists
        
        def __str__(self):
            if self.T == math.inf:
                return "SUCCESS"
            if self.T == -math.inf:
                return "FAIL"
            return "(" + str(self.T) + ", " + str(self.T_i) + ", " + str(self.W) + ", " + str(self.L) + ")"
        
        def __hash__(self):
            if str(self) == "SUCCESS":
                return hash("SUCCESS")
            if str(self) == "FAIL":
                return hash("FAIL")
            tup = [self.T]
            tup.extend(self.T_i)
            tup.append(self.W)
            tup.append(self.L)
            return hash(tuple(tup))
    

class Distribution:
    '''This class represents a discrete distribution (PMF).'''
    def __init__(self, dist_list, is_PMF=True):
        self.dist_list = dist_list  # any pair is of the form (probability, time) and it is assumed that the parameter dist_list is
                                    # sorted by times and that the probabilities sum to 1
        self.is_PMF = is_PMF        # True if dist_list is PMF, False if dist_list is CDF
    
    def PMF_to_CDF(self):
        '''assumed that is_PMF is True, create the CDF compatible with this PMF'''
        cdf = Distribution([], False)
        sum_probs = 0
        for i in range(len(self.dist_list)):
            sum_probs += self.dist_list[i][0]
            cdf.dist_list.append((sum_probs, self.dist_list[i][1]))
        return cdf
    
    def CDF_to_PMF(self):
        '''assumed that is_PMF is False, create the PMF compatible with this CDF'''
        pmf = Distribution([(self.dist_list[0][0], self.dist_list[0][1])], True)
        for i in range(1, len(self.dist_list)):
            pmf.dist_list.append((self.dist_list[i][0] - self.dist_list[i - 1][0], self.dist_list[i][1]))
        return pmf

    def compute_probability(self, t):
        '''compute the probability of t in this distribution'''
        if self.is_PMF:
            for pair in self.dist_list:
                if pair[1] == t:
                    return pair[0]
            return 0.0
        else: # the distribution is a CDF
            if  t < self.dist_list[0][1]:
                return 0.0
            for i in range(1, len(self.dist_list)):
                if t < self.dist_list[i][1]:
                    return self.dist_list[i-1][0]
            return 1.0

    def E(self):
        '''compute the expectation of a PMF'''
        ret = 0.0
        for (p, t) in self.dist_list:
            ret += p * t
        return ret

    def __str__(self):
        return str(self.dist_list)

    def sample(self):
        '''sample a value from the distribution'''
        r = random.random()
        if self.is_PMF:
            cummulative_probability = 0.0
            for (p, t) in self.dist_list:
                cummulative_probability += p
                if r < cummulative_probability:
                    return t
        else: # sample from a CDF
            for (p, t) in self.dist_list:
                if r < p:
                    return t


class Process:
    '''This class represents a process in the IPAE problem.'''

    serial_number = 0

    def __init__(self, m, d, H):
        self.serial_number = Process.serial_number
        Process.serial_number += 1
        self.m = m                  # Distribution over time of termination, it is assumed that there is a positive probability to
                                    # terminate after the last optional deadline (m[-1][1] > d[-1][1])
        self.M = m.PMF_to_CDF()     # the CDF of m
        self.d = d                  # Distribution over time of deadline
        self.D = d.PMF_to_CDF()     # the CDF of d
        self.H = H                  # tuple of actions from __B
    
    @staticmethod
    def reset_serial_number():
        Process.serial_number = 0

    def __str__(self):
        ret = "m" + str(self.serial_number) + ": " + str(self.m) + ", "
        ret += "M" + str(self.serial_number) + ": " + str(self.M) + ", "
        ret += "d" + str(self.serial_number) + ": " + str(self.d) + ", "
        ret += "D" + str(self.serial_number) + ": " + str(self.D) + ", "
        ret += "H" + str(self.serial_number) + ": ("
        for b in self.H:
            ret += str(b) + ", "
        return ret[:-2] + ")"


class BaseLevelAction:
    '''This class represents a base-level action in the IPAE problem.'''
    
    serial_number = 0

    def __init__(self, duration, deadline):
        self.serial_number = BaseLevelAction.serial_number
        BaseLevelAction.serial_number += 1
        self.duration = duration
        self.deadline = deadline
    
    @staticmethod
    def reset_serial_number():
        BaseLevelAction.serial_number = 0

    def __str__(self):
        return "(b_" + str(self.serial_number) + ", " + str(self.duration) + ")"