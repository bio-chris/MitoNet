"""

https://www.analyticsvidhya.com/blog/2018/05/24-ultimate-data-science-projects-to-boost-your-knowledge-and-skills/

first challenge: predict class based on quantitative measurements of flowers

----

5. Number of Instances: 150 (50 in each of three classes)

6. Number of Attributes: 4 numeric, predictive attributes and the class

7. Attribute Information:
   1. sepal length in cm
   2. sepal width in cm
   3. petal length in cm
   4. petal width in cm
   5. class:
      -- Iris Setosa
      -- Iris Versicolour
      -- Iris Virginica

----

"""

import warnings
warnings.simplefilter("ignore", FutureWarning)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


data_path = "datasets/irisdata.txt"

#transfer data from text file into pandas dataframe

data = pd.read_csv(data_path, sep=",", header=None)
data.columns = ["sepal length", "sepal width", "petal length", "petal width", "class"]



#data visualization

#sb.pairplot(data, hue="class")
#plt.show()

# machine learning

"""

predict classification: supervised machine learning problem 

methods to try:

k-nearest neighbors
logistic regression 
naive bayes classifier
support vector machines
decision trees / random forests

"""

# data preparation

#from sklearn.datasets import load_iris
#iris = load_iris()

#x = iris.data
#y = iris.target


x = data.drop("class", axis=1)
y = data["class"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)



# k-nearest neighbors

# n_neighbors chosen based on k_range loop
#"""
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

pred = knn.predict(x_test)


# finding the right n_neighbors value by plotting testing accuracy against n_neighbors

k_range = range(1,50)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    scores.append(accuracy_score(y_test, y_pred))

plt.plot(k_range, scores)
#plt.show()

#"""

# logistic regression
"""
lr = LogisticRegression()
lr.fit(x_train, y_train)

pred = lr.predict(x_test)
"""

# naive bayes
"""
nb = GaussianNB()
nb.fit(x_train, y_train)

pred = nb.predict(x_test)
"""

# support vector machines
"""
svm = SVC(gamma='auto')
svm.fit(x_train, y_train)

pred = svm.predict(x_test)
"""

# decision tree
"""
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)

pred = dt.predict(x_test)
"""
# accuracy testing

print(accuracy_score(y_test, pred))
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))



