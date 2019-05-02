import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import numpy as np


from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

#print(cancer['DESCR'])

df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])

print(df.head())

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(df)

scaled_data = scaler.transform(df)

# PCA

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pca.fit(scaled_data)

x_pca = pca.transform(scaled_data)

#print(scaled_data.shape)

plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'],cmap='plasma')
plt.xlabel('First PCA')
plt.ylabel('Second PCA')
plt.show()