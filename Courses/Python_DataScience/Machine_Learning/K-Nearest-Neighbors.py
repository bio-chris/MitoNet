import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


df = pd.read_csv('Classified Data',index_col=0)

print(df.head())

from sklearn.preprocessing import StandardScaler

######
# Prior processing (standardizing) before running KMeans Nearest Neighbors on the data

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()

#Compute the mean and std to be used for later scaling.
scaler.fit(df.drop('TARGET CLASS',axis=1))

# Perform standardization by centering and scaling
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))

#print(scaled_features)

df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])

print(df_feat.head())

######

from sklearn.model_selection import  train_test_split

x= df_feat
y = df['TARGET CLASS']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=34)

knn.fit(x_train, y_train)

pred = knn.predict(x_test)


from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


# "Elbow method"
# Error Rate vs K Value (checking which value for n_neighbors yields the lowest error rate)
error_rate = []

for i in range(1,40):

    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)

    pred_i = knn.predict(x_test)
    error_rate.append(np.mean(pred_i != y_test))


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
#plt.show()
"""

# Exercise

df = pd.read_csv('KNN_Project_Data')

#print(df.head())

#sb.pairplot(df,hue='TARGET CLASS')
#sb.plt.show()

from sklearn.preprocessing import StandardScaler

######
# Prior processing (standardizing) before running KMeans Nearest Neighbors on the data

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()

#Compute the mean and std to be used for later scaling.
scaler.fit(df.drop('TARGET CLASS',axis=1))

# Perform standardization by centering and scaling
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))

#print(scaled_features)

df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])

#print(df_feat.head())

######

from sklearn.model_selection import  train_test_split

x= df_feat
y = df['TARGET CLASS']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=39)

knn.fit(x_train, y_train)

pred = knn.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))



# "Elbow method"
# Error Rate vs K Value (checking which value for n_neighbors yields the lowest error rate)
error_rate = []

for i in range(1,40):

    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)

    pred_i = knn.predict(x_test)
    error_rate.append(np.mean(pred_i != y_test))


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
plt.show()

"""
