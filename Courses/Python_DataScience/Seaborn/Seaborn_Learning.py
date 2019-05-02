
import seaborn as sns

# Distribution plots

"""
tips = sns.load_dataset('tips')

#print(tips.head())

#sns.distplot(tips['total_bill'])

# distribution histogram with (kde, kernel density estimation)
#sns.plt.show()

#sns.distplot(tips['total_bill'], kde=False)
#sns.plt.show()

#sns.distplot(tips['total_bill'], kde=False,bins=30)
#sns.plt.show()

# correlation plot
#sns.jointplot(x='total_bill',y='tip',data=tips)
#sns.plt.show()

# different representation of correlation plot
# 'reg' adds a linear regression to the data
#sns.jointplot(x='total_bill',y='tip',data=tips,kind='reg')
#sns.plt.show()

#sns.jointplot(x='total_bill',y='tip',data=tips,kind='kde')
#sns.plt.show()

# pairwise relationship of data checked (pairplot)

#sns.pairplot(tips,hue='sex',palette='coolwarm')
#sns.plt.show()

# rugplot

#sns.rugplot(tips['total_bill'])
#sns.plt.show()

# Code from juypter notebook

# Don't worry about understanding this code!
# It's just for the diagram below


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Create dataset
dataset = np.random.randn(25)

# Create another rugplot
sns.rugplot(dataset);

# Set up the x-axis for the plot
x_min = dataset.min() - 2
x_max = dataset.max() + 2

# 100 equally spaced points from x_min to x_max
x_axis = np.linspace(x_min, x_max, 100)

# Set up the bandwidth, for info on this:
url = 'http://en.wikipedia.org/wiki/Kernel_density_estimation#Practical_estimation_of_the_bandwidth'

bandwidth = ((4 * dataset.std() ** 5) / (3 * len(dataset))) ** .2

# Create an empty kernel list
kernel_list = []

# Plot each basis function
for data_point in dataset:
    # Create a kernel for each point and append to list
    kernel = stats.norm(data_point, bandwidth).pdf(x_axis)
    kernel_list.append(kernel)

    # Scale for plotting
    kernel = kernel / kernel.max()
    kernel = kernel * .4
    plt.plot(x_axis, kernel, color='grey', alpha=0.5)

plt.ylim(0, 1)

#sns.plt.show()

# To get the kde plot we can sum these basis functions.

# Plot the sum of the basis function
sum_of_kde = np.sum(kernel_list,axis=0)

# Plot figure
fig = plt.plot(x_axis,sum_of_kde,color='indianred')

# Add the initial rugplot
sns.rugplot(dataset,c = 'indianred')

# Get rid of y-tick marks
plt.yticks([])

# Set title
plt.suptitle("Sum of the Basis Functions")

sns.plt.show()
"""

# Categorical plots (non-numerical data)
"""
import numpy as np

tips = sns.load_dataset('tips')

# estimator can be used to visualize other statistical values
#sns.barplot(x='sex',y='total_bill',data=tips,estimator=np.std)
#sns.plt.show()

# counting number of occurrences
#sns.countplot(x='sex',data=tips)
#sns.plt.show()

# boxplot

#sns.boxplot(x='day',y='total_bill',data=tips)
#sns.plt.show()

# violin plot

#sns.violinplot(x='day',y='total_bill',data=tips)
#sns.plt.show()

# strip plot

#sns.stripplot(x='day',y='total_bill',data=tips,jitter=True)
#sns.plt.show()

# swarm plot (combination of violin and strip plot)

#sns.swarmplot(x='day',y='total_bill',data=tips)
#sns.plt.show()

# factor plots

#sns.factorplot(x='day',y='total_bill',data=tips,kind='bar')
#sns.plt.show()

"""

# Matrix / Heatmap plots
"""
tips = sns.load_dataset('tips')
flights = sns.load_dataset('flights')

# prerequisite for matrix plots / heat maps is to bring data in matrix form

tc = tips.corr()

#sns.heatmap(tc,annot=True,cmap='coolwarm')
#sns.plt.show()

fc = flights.pivot_table(index='month',columns='year',values='passengers')
#print(fc)

#sns.heatmap(fc,cmap='magma')
# The line below rotates the labels next to y axis
#sns.plt.yticks(rotation='horizontal')
#
#sns.plt.show()

sns.clustermap(fc)
sns.plt.show()
"""

# Regression Plots
"""
tips = sns.load_dataset('tips')

sns.lmplot(x='total_bill',y='tip',data=tips, hue='sex',markers=['o','v'])
sns.plt.show()
"""

# Grids
"""
iris = sns.load_dataset('iris')

#sns.pairplot(iris)

# yields empty grids

g =sns.PairGrid(iris)
g.map_diag(sns.distplot)
g.map_upper(sns.plt.scatter)
g.map_lower(sns.kdeplot)

tips = sns.load_dataset('tips')

g = sns.FacetGrid(data=tips,col='time',row='smoker')
g.map(sns.distplot, 'total_bill')

sns.plt.show()
"""

# Style and Colour

tips = sns.load_dataset('tips')

sns.set_style('whitegrid')
sns.countplot(x='sex', data=tips)
#sns.plt.figure(figsize=(12,3))
sns.despine()
sns.plt.show()
