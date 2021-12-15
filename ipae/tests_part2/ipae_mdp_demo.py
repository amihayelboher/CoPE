import random

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