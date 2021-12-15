import matplotlib.pyplot as plt
import numpy as np
import random


mu = 25 - random.randint(-24, 24)
sigma = (mu / 10) + (mu / 10) * random.random()
s = np.random.normal(mu, sigma, 1000)

count, xs, ignored = plt.hist(s, 10, normed=True)
ys = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(xs - mu)**2 / (2 * sigma**2))
ys /= sum(ys)



# print(xs)
# print(ys)
# print(sum(ys))
plt.plot(xs, ys, linewidth=3, color='y')
plt.show()






# import matplotlib.pyplot as plt
# import numpy as np
# import scipy.stats as stats
# import math

# mu = 0
# variance = 10
# sigma = math.sqrt(variance)
# x = np.linspace(mu - sigma, mu + sigma, 10)
# # x = [val - (val % 1) + 1 for val in x]
# # x[-1] = 51
# y = stats.norm.pdf(x, mu, sigma)
# y = y / sum(y)
# x = [25 + val for val in x]
# plt.plot(x, y)
# # plt.plot(x, stats.norm.pdf(x, mu, sigma))
# plt.show()