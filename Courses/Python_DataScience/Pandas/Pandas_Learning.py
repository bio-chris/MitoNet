import numpy as np
import pandas as pd

# Series

"""


labels = ['a', 'b', 'c']
my_data = [10,20,30]
arr = np.array(my_data)
d = {'a':10, 'b':20, 'c':30}

print(pd.Series(data=my_data))

print(pd.Series(data=my_data,index=labels)) # "pd.Series(my_data, labels)" also possible

print(pd.Series(arr, labels))

print(pd.Series(d))


ser1 = pd.Series([1,2,3,4], ['USA', 'Germany', 'USSR', 'Japan'])

ser2 = pd.Series([1,2,5,4], ['USA', 'Germany', 'Italy', 'Japan'])


print(ser1['USA'])

print(ser1+ser2)


# DataFrames

from numpy.random import randn

np.random.seed(101)

df = pd.DataFrame(randn(5,4),['A', 'B', 'C', 'D', 'E'], ['W', 'X', 'Y', 'Z'])

print(df)

print(df['X'])

print(df[['X', 'Z']])

df['new'] = df['W'] + df['Y']

print(df)

print(df.drop('E'))

# Selecting rows

print(df.loc['A'])

print(df.iloc[0])

print(df.loc['B','Y'])

print(df.loc[['A', 'B'], ['W', 'Y']])


print(df>0)

print(df['W']>0)

print((df['W']>0) | (df['Y']>1))



# Index Levels

outside = ['G1','G1','G1','G2','G2','G2']
inside = [1,2,3,1,2,3]
hier_index = list(zip(outside,inside))
hier_index = pd.MultiIndex.from_tuples(hier_index)

# Multilevel DataFrame
df = pd.DataFrame(randn(6,2), hier_index, ['A', 'B'])

print(df)

print(df.loc['G1'].loc[1])


# Missing Data

d = {'A':[1,2,np.nan], 'B':[5,np.nan,np.nan], 'C':[1,2,3]}

df = pd.DataFrame(d)

print(df)

# drops (removes) all rows with NaN
print(df.dropna())

# drops all columns with NaN
print(df.dropna(axis=1))

# drops rows that have more than the specified number of NaN values (threshold)
print(df.dropna(thresh=2))

# replace all NaN values with specified value
print(df.fillna('fill'))

print(df['A'].fillna(value=df['A'].mean()))


# Groupby

# groupby allows you to group together rows based of a column and perform an aggregate function on them


data = {'Company':['GOOG','GOOG','MSFT','MSFT','FB','FB'],
       'Person':['Sam','Charlie','Amy','Vanessa','Carl','Sarah'],
       'Sales':[200,120,340,124,243,350]}

df = pd.DataFrame(data)

print(df)
# group by company column and setting equal to variable
byComp = df.groupby('Company')

# performing different calculations on grouped company values
print(byComp.mean())

print(byComp.sum())

print(byComp.sum().loc['FB'])

# same result as above but all code is in one line
print(df.groupby('Company').sum().loc['FB'])

print(df.groupby('Company').count())

# get statistical information from table
print(df.groupby('Company').describe())


# Merging, Joining and Concatenating

df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3'],
                        'C': ['C0', 'C1', 'C2', 'C3'],
                        'D': ['D0', 'D1', 'D2', 'D3']},
                        index=[0, 1, 2, 3])

df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                        'B': ['B4', 'B5', 'B6', 'B7'],
                        'C': ['C4', 'C5', 'C6', 'C7'],
                        'D': ['D4', 'D5', 'D6', 'D7']},
                         index=[4, 5, 6, 7])

df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
                        'B': ['B8', 'B9', 'B10', 'B11'],
                        'C': ['C8', 'C9', 'C10', 'C11'],
                        'D': ['D8', 'D9', 'D10', 'D11']},
                        index=[8, 9, 10, 11])

# Concatenation

print(pd.concat([df1,df2,df3]))

print(pd.concat([df1,df2,df3], axis=1))

# Merging

left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})

right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                      'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']})

# merging based on key column
print(pd.merge(left,right, how='inner', on='key'))

# Joining

left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                     'B': ['B0', 'B1', 'B2']},
                      index=['K0', 'K1', 'K2'])

right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                    'D': ['D0', 'D2', 'D3']},
                      index=['K0', 'K2', 'K3'])


print(left.join(right))


# Operations

df = pd.DataFrame({'col1':[1,2,3,4],'col2':[444,555,666,444],'col3':['abc','def','ghi','xyz']})

print(df.head())

# Finding unique values in dataframe

print(df['col2'].unique())

# prints the amount of unique values
print(len(df['col2'].unique()))
print(df['col2'].nunique())

# prints how times each values occurred
print(df['col2'].value_counts())

# Selecting Data

print(df['col1']>2)

# Apply Methods

def times2(x):
    return x*2

print(df['col1'].sum())

# will apply the function to each value in the column
print(df['col1'].apply(times2))

print(df['col3'].apply(len))

print(df['col2'].apply(lambda x:x*2))

# Removing columns

print(df.drop('col1',axis=1))

# Sorting a data frame

print(df.sort_values('col2'))

# Pivot table

data = {'A':['foo','foo','foo','bar','bar','bar'],
     'B':['one','one','two','two','one','one'],
       'C':['x','y','x','y','x','y'],
       'D':[1,3,2,5,4,1]}

df = pd.DataFrame(data)

print(df)

print(df.pivot_table(values='D', index=['A','B'], columns=['C']))
"""

# Data Input and Output

# CSV
df = pd.read_csv('example.csv')

print(df)

# Excel

df = pd.read_excel('Excel_Sample.xlsx', sheetname='Sheet1')

print(df)

# new excel file

# df.to_excel('Excel_Sample2.xlsx', sheet_name='NewSheet')

# HTML

#df = pd.read_html('http://www.fdic.gov/bank/individual/failed/banklist.html')

#print(df[0].head())

# SQL

from sqlalchemy import create_engine

engine = create_engine('sqlite:///:memory:')

df.to_sql('my_table', engine)

sqldf = pd.read_sql('my_table', con=engine)

print(sqldf)


