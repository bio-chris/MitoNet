
# Learning section

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

df = pd.read_csv('USA_Housing.csv')

#print(df.head())

#sb.pairplot(df)

#sb.distplot(df['Price'])

#sb.heatmap(df.corr())
#sb.plt.show()


X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]

y = df['Price']

from sklearn.model_selection import train_test_split

# test_size is % of dataset that will be allocated
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train, y_train)

#print(lm.intercept_)

cdf = pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])
#print(cdf)


#from sklearn.datasets import load_boston
#boston = load_boston()
#print(boston)


# predictions

predictions = lm.predict(X_test)

#plt.scatter(y_test, predictions)
#plt.show()

#sb.distplot((y_test-predictions))
#sb.plt.show()

from sklearn import metrics

metrics.mean_absolute_error(y_test,predictions)
"""

# Exercise

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

customers = pd.read_csv('Ecommerce Customers')

#print(customers.head())

# data exploration

#sb.jointplot(x='Time on Website',y='Yearly Amount Spent', data=customers)
#sb.jointplot(x='Time on App',y='Yearly Amount Spent', data=customers)
#sb.jointplot(x='Time on App',y='Length of Membership', data=customers, kind='hex')
#sb.pairplot(customers)
#sb.lmplot(x='Length of Membership',y='Yearly Amount Spent', data=customers)

#sb.plt.show()

# training and testing data

#print(customers.columns)
x = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]

y = customers['Yearly Amount Spent']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

# training the model

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x_train,y_train)

cdf = pd.DataFrame(lr.coef_, x.columns,columns=['Coefficient'])
#print(cdf)

predictions = lr.predict(x_test)

#plt.scatter(y_test, predictions)
#plt.show()

from sklearn import metrics

print(metrics.mean_absolute_error(y_test,predictions))
print(metrics.mean_squared_error(y_test,predictions))
print(np.sqrt(metrics.mean_squared_error(y_test,predictions)))

#sb.distplot((y_test-predictions))
#sb.plt.show()

print(cdf)


