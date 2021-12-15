import matplotlib.pyplot as plt
import pandas as pd


success = [0.583, 0.052, 0.55, 0.572, 0, 0, 0.559, 0.591, 0.51, 0.557, 0.542, 0.609, 0.605, 0.575, 0.292, 0.248,
                0.254, 0.605]
runtime = [0.662, 0.299, 0, 0, 0, 0, 0.048, 0.05, 1.862, 1.836, 1.756, 7.05, 0.77, 15.338, 0.902, 2.357, 10.869, 3.689]
algorithm = ["dp", "rand", "mpp", "de+mpp", "rr", "de+rr", "bgs", "de+bgs", "dda", "de+dda", "r-mlet+bgs", "mlet+dp",
       "mlet+bgs", "mlet+dda", "mcts+10", "mcts+100", "mcts+500", "2-bnd+bgs"]
colors = {"dp": 'black', "rand": 'gray', "mpp": 'blueviolet', "de+mpp": 'firebrick', "rr": 'bisque',
          "de+rr": 'moccasin', "bgs": 'red', "de+bgs": 'chartreuse', "dda": 'navy', "de+dda": 'orange',
          "r-mlet+bgs": 'olivedrab', "mlet+dp": 'lightseagreen', "mlet+bgs": 'blue', "mlet+dda": 'fuchsia',
          "mcts+10": 'springgreen', "mcts+100": 'paleturquoise', "mcts+500": 'purple', "2-bnd+bgs": 'yellow'}
df = pd.DataFrame(dict(success=success, runtime=runtime, algorithm=algorithm))

fig, ax = plt.subplots()
ax.scatter(df['runtime'], df['success'], c=df['algorithm'].map(colors))
plt.show()