import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

"""
df = pd.read_csv('kyphosis.csv')

#print(df.head())
#print(df.info())

#sb.pairplot(df, hue='Kyphosis')
#sb.plt.show()

from sklearn.model_selection import train_test_split

x = df.drop('Kyphosis',axis=1)

y = df['Kyphosis']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

dtree.fit(x_train,y_train)

predictions = dtree.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=200)

rfc.fit(x_train,y_train)

rfc_pred = rfc.predict(x_test)

print(confusion_matrix(y_test, rfc_pred))
print(classification_report(y_test, rfc_pred ))
"""

df = pd.read_csv("loan_data.csv")

#print(df.head())
#print(df.info())
#print(df.describe())

#df['fico'].hist(bins=30,edgecolor='black')

#df[df['credit.policy']==1]['fico'].hist(bins=30,edgecolor='black')
#df[df['credit.policy']==0]['fico'].hist(bins=30,edgecolor='black',color='red')

#df[df['not.fully.paid']==1]['fico'].hist(bins=30,edgecolor='black')
#df[df['not.fully.paid']==0]['fico'].hist(bins=30,edgecolor='black',color='red',alpha=0.5)

#sb.plt.show()

#sb.countplot(data=df, y='purpose', hue='not.fully.paid')

#sb.jointplot(x='fico',y='int.rate',data=df)

#sb.lmplot(x='fico',y='int.rate',data=df,col='not.fully.paid')
#sb.plt.show()

cat_feats = ['purpose']

final_data = pd.get_dummies(df,columns=cat_feats,drop_first=True)
#print(final_data.head())

from sklearn.model_selection import train_test_split

x = final_data.drop('not.fully.paid',axis=1)

y = final_data['not.fully.paid']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

dtree.fit(x_train,y_train)

predictions = dtree.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=200)

rfc.fit(x_train,y_train)

rfc_pred = rfc.predict(x_test)

print(confusion_matrix(y_test, rfc_pred))
print(classification_report(y_test, rfc_pred ))