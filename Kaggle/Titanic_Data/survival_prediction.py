import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sb


# Read the Data
#########


df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

test_passengerID = df_test["PassengerId"]

#########


# Data Exploration
#########

# checks if any values are NaN (True means NaN)
#print(df_train.isnull())

# reveals correlation between class and age (low class like 3 have lower median age)
#sb.boxplot(x='Pclass',y='Age',data=df_train)
#plt.show()

#sb.countplot(x='Survived',hue='Sex', data=df_train)
#plt.show()


#########


# Data cleaning
#########

# removing irrelevant data
l = ["PassengerId", "Name", "Ticket", "Cabin"]
df_train = df_train.drop(l, axis=1)
df_test = df_test.drop(l, axis=1)

# converting categorical data (male or female) into binary data (1 or 0)
sex = pd.get_dummies(df_train['Sex'],drop_first=True)
df_train["Sex"] = sex

#test data
sex_test = pd.get_dummies(df_test['Sex'],drop_first=True)
df_test["Sex"] = sex_test

# converting categorical data (place from which one left)
embark = pd.get_dummies(df_train['Embarked'],drop_first=True)

df_train = pd.concat([df_train,embark],axis=1)
df_train = df_train.drop("Embarked", axis=1)

# same procedure as above, with test data
embark_test = pd.get_dummies(df_test['Embarked'],drop_first=True)

df_test = pd.concat([df_test,embark_test],axis=1)
df_test = df_test.drop("Embarked", axis=1)


# more advanced (or precise) method of interpolation, taking into account the correlation between class and age
def impute_age(cols):

    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):

        if Pclass == 1:
            return np.mean(df_train['Age'][df_train['Pclass']==1])
        elif Pclass ==2:
            return np.mean(df_train['Age'][df_train['Pclass']==2])
        else:
            return np.mean(df_train['Age'][df_train['Pclass']==3])

    else:

        return Age

df_train['Age'] = df_train[['Age','Pclass']].apply(impute_age,axis=1)

df_test['Age'] = df_test[['Age','Pclass']].apply(impute_age,axis=1)



# removes any rows with nan values
#df_train.dropna(inplace=True)

#df_test.dropna(inplace=True)


#exit()

#df_train.dropna(inplace=True)




# TRAINING DATA
# Machine learning section
######

# Logistic regression

x = df_train.drop('Survived',axis=1)
y = df_train['Survived']

print(x.head)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LogisticRegression

"""

parameter tuning in logistic regression 

C
penalty
random state

"""

logmodel = LogisticRegression()

logmodel.fit(x_train,y_train)

predictions = logmodel.predict(x_test)

#print(len(predictions))
#print(len(passenger_id))

#predictions_table = pd.DataFrame(columns=["PassengerId", "Survived"], index=[passenger_id, predictions])
#print(predictions_table)


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print(classification_report(y_test,predictions))
#print(confusion_matrix(y_test,predictions))



#TEST DATA