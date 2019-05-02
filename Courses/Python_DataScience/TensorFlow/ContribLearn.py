# using TensorFlow SKFlow (Sci-Kit Learn interface)
"""
from sklearn.datasets import load_iris

iris = load_iris()

# print(iris)

X = iris["data"]
y = iris["target"]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

import tensorflow.contrib.learn as learn
import tensorflow as tf

# dimensions specifies number of features
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

# DNN = deep neural network, n_classes specifies the number of classes (in this case 3 iris species)
# hidden units, how many nodes per layer
classifier = learn.DNNClassifier(hidden_units=[10, 20, 10], n_classes=3, feature_columns=feature_columns)

classifier.fit(x_train, y_train, steps=300, batch_size=32)

# classifier.predict is generator object
iris_predictions = classifier.predict(x_test)

new_iris_predictions = []

for i in iris_predictions:
    new_iris_predictions.append(i)

from sklearn.metrics import confusion_matrix, classification_report

print(classification_report(y_test, new_iris_predictions))
"""

# Project exercise

import pandas as pd

df = pd.read_csv("bank_note_data.csv")

#print(df.head())

import seaborn as sb
import matplotlib.pyplot as plt

#sb.countplot(df["Class"])

#sb.pairplot(df, hue="Class")
#plt.show()

# Data Preparation

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(df.drop("Class", axis=1))

scaled_df = pd.DataFrame(scaler.fit_transform(df.drop("Class", axis=1)), columns=df.columns[:-1])
#print(scaled_df.head())

X = scaled_df.as_matrix()
y = df["Class"].as_matrix()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

import tensorflow.contrib.learn as learn
import tensorflow as tf

# dimensions specifies number of features
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

# DNN = deep neural network, n_classes specifies the number of classes (in this case 3 iris species)
# hidden units, how many nodes per layer
classifier = learn.DNNClassifier(hidden_units=[10, 20, 10], n_classes=3, feature_columns=feature_columns)

classifier.fit(x_train, y_train, steps=200, batch_size=20)

# classifier.predict is generator object
predictions = classifier.predict(x_test)

new_predictions = []

for i in predictions:
    new_predictions.append(i)

from sklearn.metrics import confusion_matrix, classification_report

print(classification_report(y_test, new_predictions))


