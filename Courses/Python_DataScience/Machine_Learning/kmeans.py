import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

"""

from sklearn.datasets import make_blobs

data = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1.8,
                  random_state=101)

#print(data)

#plt.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')
#plt.show()

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4)

kmeans.fit(data[0])

kmeans.labels_

"""

# Project

df = pd.read_csv('College_Data', index_col=0)

#print(df.head())

#print(df['Private'])

#sb.lmplot(x='Grad.Rate',y='Room.Board', data=df ,hue='Private', fit_reg=False)

#sb.lmplot(y='F.Undergrad',x='Outstate', data=df ,hue='Private', fit_reg=False)

#g = sb.FacetGrid(df, hue='Private')
#g.map(plt.hist, 'Outstate',edgecolor='black',bins=30)
#sb.plt.show()

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2)

kmeans.fit(df.drop('Private', axis=1))

print(kmeans.cluster_centers_)

def converter(private):
    if private == 'Yes':
        return 1
    else:
        return 0

df['Cluster'] = df['Private'].apply(converter)

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(df['Cluster'],kmeans.labels_))
print(classification_report(df['Cluster'],kmeans.labels_))