import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0,100)
y = x*2
z = x**2


"""
fig = plt.figure()


axes = fig.add_axes([0.1,0.1,0.8,0.8])
axes2 = fig.add_axes([0.2,0.5,0.3,0.3])
axes.plot(x,z)
axes2.plot(x,y)

axes.set_xlabel("x")
axes.set_ylabel("z")

axes2.set_xlabel("x")
axes2.set_ylabel("y")
axes2.set_xlim([20,22])
axes2.set_ylim([30,50])
"""

fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(10,2))

axes[0].plot(x,y,color="blue",linestyle="--")
axes[1].plot(x,z,color="red")


plt.show()


"""
ax.plot(x,y,color='blue',linewidth=1,alpha=1,linestyle="--",marker='*',markersize=10,
        markerfacecolor='red', markeredgewidth=1)
"""


"""
# plot inside plot / manual positioning of plot 
fig = plt.figure()

axes1 = fig.add_axes([0.1,0.1,0.8,0.8])
axes2 = fig.add_axes([0.2,0.5,0.4,0.3])

axes1.plot(x,y)
axes2.plot(y,x)

plt.show()
"""
