import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

"""

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

#print(cancer['DESCR'])

df_feat = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])

#print(df_feat.head())

from sklearn.model_selection import train_test_split

x = df_feat
y = cancer['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

from sklearn.svm import SVC

model = SVC()

model.fit(x_train,y_train)

predictions = model.predict(x_test)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

# GridSearchCV takes in a dictionary that describes parameters that should be tried in a model to train
# GridSearchCV is mandatory when using Support Vector Machines
from sklearn.model_selection import GridSearchCV

# testing different values
param_grid = {'C':[0.1,1,10,100,1000], 'gamma':[1,0.1,0.01,0.001,0.0001]}

grid = GridSearchCV(SVC(),param_grid,verbose=3)

grid.fit(x_train,y_train)

print(grid.best_params_)
print(grid.best_estimator_)

grid_predictions = grid.predict(x_test)

print(classification_report(y_test,grid_predictions))
print(confusion_matrix(y_test,grid_predictions))

"""

# Project

iris = sb.load_dataset('iris')

#print(iris.head())

#sb.pairplot(iris, hue='species')

#print(len(file[file['Year']==2013]['JobTitle'].unique()))

#sb.kdeplot(iris[iris['species']=='setosa']['sepal_width'],iris[iris['species']=='setosa']['sepal_length'])
#sb.plt.show()

from sklearn.model_selection import train_test_split

x = iris.drop('species',axis=1)
y = iris['species']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

from sklearn.svm import SVC

model = SVC()

model.fit(x_train,y_train)

predictions = model.predict(x_test)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))