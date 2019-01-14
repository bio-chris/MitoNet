
import matplotlib.pyplot as plt
import numpy as np

# Command to show generated plots
#plt.show()

x = np.linspace(0,5,11)
y = x**2

#print(x,y)
"""
# Functional Plot method

plt.plot(x,y)

# plot labeling
plt.xlabel("X Label")
plt.ylabel("Y Label")
plt.title("Title")

#plt.show()

# Multiplots

# (number of rows, number of columns, number of plot one is referring to)
plt.subplot(1,2,1)
plt.plot(x,y,'r')

plt.subplot(1,2,2)
plt.plot(y,x,'b')

#plt.show()
"""
# Object Oriented method

# figure object
"""
fig = plt.figure()

axes = fig.add_axes([0.1,0.1,0.8,0.8])

axes.plot(x,y)
axes.set_xlabel("X Label")
axes.set_ylabel("Y Label")
axes.set_title("Title")

plt.show()
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

# multiplots
"""
fig,axes = plt.subplots(nrows=1,ncols=2)


# iterate through axes array
for current_ax in axes:
    current_ax.plot(x,y)


axes[0].plot(x,y)
axes[0].set_title("First")
axes[1].plot(y,x)
axes[1].set_title("Second")

plt.show()
"""

# figure size and dpi

# dpi sets thickness of graph // usually is left as default
"""
fig, axes = plt.subplots(nrows=2,ncols=1,figsize=(8,2))

axes[0].plot(x,y)
axes[1].plot(y,x)

plt.tight_layout()

plt.show()

# save a figure

fig.savefig('my_file.png')

# add a figure legend

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])
ax.plot(x,x**2,label='x squared')
ax.plot(x,x**3,label='x cubed')

ax.legend()

plt.show()
"""

# Plot appearance

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

# color= blue, orange, red, etc. or RGB Hex code using (eg. #FF8C00)
# alpha controls transparency
ax.plot(x,y,color='blue',linewidth=1,alpha=1,linestyle="--",marker='*',markersize=10,
        markerfacecolor='red', markeredgewidth=1)

# control over axes appearance
ax.set_xlim([0,1])
ax.set_ylim([0,2])

plt.show()


