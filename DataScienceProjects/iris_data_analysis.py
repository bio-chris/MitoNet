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

import pandas as pd

data_path = "iris_data/irisdata.txt"


# first task: transfer data from text file into pandas dataframe

data = pd.read_csv(data_path, sep=",", header=None)
data.columns = ["sepal length", "sepal width", "petal length", "petal width", "class"]

print(data)