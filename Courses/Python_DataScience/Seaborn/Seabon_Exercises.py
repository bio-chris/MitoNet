
import seaborn as sns
import matplotlib.pyplot as plt

titanic = sns.load_dataset('titanic')

#print(titanic.head())

#sns.jointplot(x='fare',y='age',data=titanic)

#sns.distplot(titanic['fare'],kde=False)

#sns.boxplot(x='class',y='age',data=titanic)

#sns.swarmplot(x='class',y='age',data=titanic)

#sns.countplot(x='sex',data=titanic)

#tc = titanic.corr()
#sns.heatmap(tc,annot=True,cmap='coolwarm', )


# yields empty grids

g = sns.FacetGrid(data=titanic,col='sex')
g.map(sns.distplot, 'age', kde=False)


sns.plt.show()