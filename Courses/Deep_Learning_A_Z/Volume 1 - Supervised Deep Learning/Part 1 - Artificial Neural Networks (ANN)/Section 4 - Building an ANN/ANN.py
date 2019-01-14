# Part 1 - Data Preprocessing


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

org_X = X

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# one hot encoder turns one column containing names of three different countries (depicted as 0,1,2) into
# three columns containing either 0 or 1 to depict which country it is

labelencoder_X_1 = LabelEncoder()
labelencoder_X_2 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])



onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

# avoiding dummy trap (three countries, if both columns are 0, third country has to be 1)
X = X[:, 1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

# for training set, first fitting then transform
X_train = sc.fit_transform(X_train)
# for test set, only transform (no fitting)
X_test = sc.transform(X_test)




# ANN
##########

# Importing keras libraries and packages

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# initialise ANN

classifier = Sequential()


#units: nr of nodes, kernel_initializer: weight initialisation, activation: activation function, input_dim: nr of
#independent variables

# adding input and first hidden layer with dropout (randomly deactivates neurons to prevent overfitting)
classifier.add(Dense(units = 6, kernel_initializer= "uniform", activation="relu", input_dim=11))
# 10% of neurons will be deactivated (p=0.1)
classifier.add(Dropout(p=0.1))

# adding second hidden layer with dropout
classifier.add(Dense(units = 6, kernel_initializer= "uniform", activation="relu"))
classifier.add(Dropout(p=0.1))


# adding the output layer
classifier.add(Dense(units = 1, kernel_initializer= "uniform", activation="sigmoid"))

# compiling the ANN
"""
#optimizer: adam (specific algorithm for stochastic gradient descent)
#loss: loss function (logarithmic loss function, called binary_crossentropy (for binary outcome))
#if more than two outcomes, use categorical_crossentropy
#metrics argument: criteria to evaluate your model
"""
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# fitting the ANN
"""
#batch size: nr of observations after to update weights
"""
#classifier.fit(X_train, y_train, batch_size=10 , nb_epoch=30)

##########


#classifier.save('my_model.h5')




from keras.models import load_model

classifier = load_model('my_model.h5')

# making predictions and evaluating model

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

####
# prediction for one single observation

# applying same transformation as to training set
new_predictions = classifier.predict(sc.transform(np.array([[0,0, 600, 1, 40, 30, 60000, 2, 1, 1, 50000]])))
new_predictions = (new_predictions > 0.5)

#print(new_predictions)
####

# Creating the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# Evaluating the ANN

"""

bias-variance tradeoff 

evaluate model performance: k-fold cross validation 

splitting training set into 10 training folds 

"""


# k-fold cross validation

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
# used for model tuning
from sklearn.model_selection import GridSearchCV


def build_classifier():

    classifier = Sequential()

    # adding input and first hidden layer
    classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu", input_dim=11))
    # adding second hidden layer
    classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu"))
    # adding the output layer
    classifier.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))

    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return classifier

"""

the adam optimizer is an extension to stochastic gradient descent 

It is used to update network weights iterative based in training data

Adam (adaptive moment estimation) 

Stochastic gradient descent maintains a single learning rate (does not change during training)


"""

# wrapping

classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, nb_epoch=5)


if __name__ == '__main__':

    # k-fold cross validation
    accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1)
    print(accuracies)


# Improving the ANN (Parameter tuning)

# dropout regularization to reduce overfitting if needed (see above!)

"""

hyperparameters are parameters that remain unchanged (weights are not hyperparameters)

examples for hyperparameters are nr of epochs, batch_size, optimizer, nr of neurons in layer 

grid search returns best selection of hyperparameters 

"""

def build_classifier(optimizer):

    classifier = Sequential()

    # adding input and first hidden layer
    classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu", input_dim=11))
    # adding second hidden layer
    classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu"))
    # adding the output layer
    classifier.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))

    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return classifier

classifier = KerasClassifier(build_fn=build_classifier)

parameters = {'batch_size': [25, 32],
              'nb_epoch': [100, 500],
              'optimizer': ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator= classifier,
                           param_grid=parameters,
                           scoring=accuracies,
                           cv=10)


grid_search = grid_search.fit(X_train, y_train)

# getting best parameters and best accuracies 
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_



