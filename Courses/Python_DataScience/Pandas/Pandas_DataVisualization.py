"""
import numpy as np
import pandas as pd
import seaborn as sns


df1 = pd.read_csv('df1',index_col=0)

df2 = pd.read_csv('df2')


#df1['A'].plot.hist(edgecolor='black')

#df2.plot.bar()

#df1.plot.line(x=df1.index,y='B')

df1.plot.scatter(x='A',y='B')

sns.plt.show()

"""

# Exercises

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df3 = pd.read_csv('df3')


#df3.plot.scatter(x='a',y='b',figsize=(12,3))
#df3['a'].plot.hist(edgecolor='black',bins=30)
#df3.plot.box(x=['c','d'])

df3['d'].plot.kde(linestyle='--')

sns.plt.show()
