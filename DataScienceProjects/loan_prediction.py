"""

https://www.analyticsvidhya.com/blog/2018/05/24-ultimate-data-science-projects-to-boost-your-knowledge-and-skills/

second challenge: loan prediction

data from https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/

----

About Company

Dream Housing Finance company deals in all home loans. They have presence across all urban, semi urban and rural areas.
Customer first apply for home loan after that company validates the customer eligibility for loan.
Problem

Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling
online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount,
Credit History and others. To automate this process, they have given a problem to identify the customers segments, those
are eligible for loan amount so that they can specifically target these customers. Here they have provided a partial
data set.

Data

Variable: Description
Loan_ID: Unique Loan ID
Gender: Male/ Female
Married: Applicant married (Y/N)
Dependents: Number of dependents
Education: Applicant Education (Graduate/ Under Graduate)
Self_Employed: Self employed (Y/N)
ApplicantIncome: Applicant income
CoapplicantIncome: Coapplicant income
LoanAmount: Loan amount in thousands
Loan_Amount_Term: Term of loan in months
Credit_History: credit history meets guidelines
Property_Area: Urban/ Semi Urban/ Rural
Loan_Status: Loan approved (Y/N)

Note:
    Evaluation Metric is accuracy i.e. percentage of loan approval you correctly predict.
    You are expected to upload the solution in the format of "sample_submission.csv"


----

"""

import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


data_train = pd.read_csv("datasets/loan_data_train.txt", sep=",")
data_test = pd.read_csv("datasets/loan_data_test.txt", sep=",")

#print(data_train.columns)

# preparing data

x = data_train.drop(["Loan_ID", "Loan_Status"], axis=1)
y = data_train["Loan_Status"]

#print(x.isnull().sum())


"""
======
16-01-19: 
was not able to find code to replace nan values with categorical labels will therefore now move on to
remove nan containing rows in data. will readdress this later

update (see below):

pandas fillnan works 

======

# converting pd series into np array of form [[]] to run imputer on and replace nan values
test = np.array(x["Gender"])

test = test.reshape(1,-1)

#print(x.dtypes)

# showing the number of nan values in data
print(x.isnull().sum())

print(x.dtypes)

from sklearn.impute import SimpleImputer

num_imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean', fill_value="constant")

cat_imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')

new_test = cat_imputer.fit_transform(test)

#print(new_test)

print(pd.Series(new_test[0]))

print(x)

print(x.isnull().sum())

"""

# using the dropna approach: biggest problem, going from 614 rows to 480 (loosing over 134 rows of data)
#print(x) #.dropna())


# replace missing values in one column only by the most frequent value in that column
#x = x.fillna(x['Label'].value_counts().index[0])

# replace missing values in all columns
x = x.apply(lambda x:x.fillna(x.value_counts().index[0]))


lab_enc = LabelEncoder()

# female or male = [0, 1]
x["Gender"] = lab_enc.fit_transform(x["Gender"])

# no or yes = [0, 1]
x["Married"] = lab_enc.fit_transform(x["Married"])

# ['0', '1', '2', '3+'] = [0,1,2,3]
x["Dependents"] = lab_enc.fit_transform(x["Dependents"])
#print(list(lab_enc.inverse_transform([0,1,2,3])))

# graduate or under graduate = [0,1]
x["Education"] = lab_enc.fit_transform(x["Education"])

# no or yes = [0, 1]
x["Self_Employed"] = lab_enc.fit_transform(x["Self_Employed"])

# ['Rural', 'Semiurban', 'Urban'] = [0,1,2]
x["Property_Area"] = lab_enc.fit_transform(x["Property_Area"])


#print(x.dtypes)

scaled = StandardScaler()
scaled_data = scaled.fit_transform(x)

#print(scaled_data)

scaled_data = pd.DataFrame(scaled_data, columns=x.columns) #, columns=df.columns[:-1])


# splitting data into train and test

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# naive bayes
"""
nb = GaussianNB()
nb.fit(x_train, y_train)

pred = nb.predict(x_test)
"""

# decision tree
"""
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)

pred = dt.predict(x_test)
"""

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
plt.show()

print(accuracy_score(y_test, pred))
#print(confusion_matrix(y_test,pred))
#print(classification_report(y_test,pred))



