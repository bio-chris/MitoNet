#"""
import pandas as pd
import numpy as np

import seaborn as sb
import matplotlib.pyplot as plt

train = pd.read_csv('titanic_train.csv')

# Data Exploration

#print(train.head())

# checks if any values are NaN (True means NaN)
#print(train.isnull())

#sb.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#sb.plt.show()

sb.set_style('whitegrid')

#sb.countplot(x='Survived',hue='Pclass', data=train)


#sb.distplot(train['Age'].dropna(),kde=False,bins=30)

#sb.countplot(x='SibSp',data=train)
#sb.plt.show()

# Cleaning Data

#sb.boxplot(x='Pclass',y='Age',data=train)
#sb.plt.show()

#print(train['Age']['Pclass']==1)

#print(np.mean(train['Age'][train['Pclass']==1]))
def impute_age(cols):

    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):

        if Pclass == 1:
            return np.mean(train['Age'][train['Pclass']==1])
        elif Pclass ==2:
            return np.mean(train['Age'][train['Pclass']==2])
        else:
            return np.mean(train['Age'][train['Pclass']==3])

    else:

        return Age


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)

#sb.heatmap(train.isnull(),yticklabels=False,cbar=False)
#sb.plt.show()

train.drop('Cabin',axis=1,inplace=True)
train.dropna(inplace=True)

# creating a dummy variable (converting a string into a variable (eg. 0 or 1))

sex = pd.get_dummies(train['Sex'],drop_first=True)

embark = pd.get_dummies(train['Embarked'],drop_first=True)

train = pd.concat([train,sex,embark],axis=1)

#print(train.head())

train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

#print(train.head())

x = train.drop('Survived',axis=1)

print(x.head())

y = train['Survived']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(x_train,y_train)

predictions = logmodel.predict(x_test)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

"""

# Exercise

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

ad_data = pd.read_csv('advertising.csv')

#print(ad_data.head())

# Exploratory Data Analysis

#sb.distplot(ad_data['Age'],kde=False, hist_kws=dict(edgecolor='k',linewidth=2))

#sb.jointplot(x='Age',y='Area Income',data=ad_data)

#sb.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_data,kind="kde")

#sb.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=ad_data)

#sb.pairplot(ad_data, hue='Clicked on Ad')
#sb.plt.show()

# Logistic regression

ad_data.drop(['Ad Topic Line','City','Country','Timestamp'],axis=1,inplace=True)

x = ad_data.drop('Clicked on Ad',axis=1)
y = ad_data['Clicked on Ad']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(x_train,y_train)

predictions = logmodel.predict(x_test)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

"""
