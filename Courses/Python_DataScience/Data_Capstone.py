"""
# 911 project

import pandas as pd
import numpy as np
import seaborn as sb



df = pd.read_csv('911.csv')

#print(df.info())


# 5 top zipcodes for 911 calls
#print(df['zip'].value_counts().nlargest(5))

# top 5 townships for 911 calls
#print(df['twp'].value_counts().nlargest(5))

# how many unique title codes are there
#print(len(df['title'].unique()))

# add new column called reason that contains substring from the title column
df['reason']=df['title'].apply(lambda x: x.split(":")[0])

# most common reason for 911 based on the reason column
#print(df['reason'].value_counts())

# seaborn countplot

#sb.countplot(x='reason', data=df)
#sb.plt.show()

# what is data type of objects in timeStamp column
#print(type(df['timeStamp'][0]))

# convert to datetime objects
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
#print(type(df['timeStamp'][0]))

# create three new columns containing hour, month and day info
df['hour']=df['timeStamp'].apply(lambda x: x.hour)
df['month']=df['timeStamp'].apply(lambda x: x.month)
df['day of week']=df['timeStamp'].apply(lambda x: x.weekday())

#print(df.head())

dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['day of week']=df['day of week'].apply(lambda x: dmap[x])

# seaborn countplot Day of Week (and month) column with the hue based off of the Reason column

#sb.countplot(x='day of week', data=df, hue='reason')
#sb.countplot(x='month', data=df, hue='reason')
#sb.plt.show()

# Group by month

byMonth = df.groupby('month').count()

#print(byMonth.head())

# plot dataframe indicating counts of calls per month

#byMonth['lat'].plot()
#sb.lmplot(x='month',y='lat',data=byMonth.reset_index())

#sb.plt.show()

# create new column called 'Date' that contains date from timeStamp column

df['Date']=df['timeStamp'].apply(lambda x: x.date())

print(df.head())

# grouby this date column with the count() aggregate and create a plot of counts of 911 calls

byDate = df.groupby('Date').count()

#byDate['lat'].plot()

# create 3 plots based on reason for 911 calls based on byDate data

#df[df['reason']=='Traffic'].groupby('Date').count()['twp'].plot()
# same line of code for Fire or EMS as reason
#sb.plt.show()

# regroup data for heatmaps

dayHour = df.groupby(by=['day of week','hour']).count()['reason'].unstack()

# generate simple heatmap
#sb.heatmap(data=dayHour)

# clustermap
#sb.clustermap(data=dayHour)

# repeat plots with month as the column

dayMonth = df.groupby(by=['day of week','month']).count()['reason'].unstack()

#sb.heatmap(data=dayMonth)
sb.clustermap(data=dayMonth)
sb.plt.show()
"""

# Finance project

from pandas_datareader import data, wb
import pandas as pd
import numpy as np
import datetime
import seaborn as sb

start = datetime.datetime(2006,1,1)
end = datetime.datetime(2016,1,1)

BAC = data.DataReader("NYSE:BAC", 'google', start, end)
CG = data.DataReader("NYSE:C", 'google', start, end)
GS = data.DataReader("NYSE:GS", 'google', start, end)
JPMC = data.DataReader("NYSE:JPM", 'google', start, end)
MS = data.DataReader("NYSE:MS", 'google', start, end)
WF = data.DataReader("NYSE:WFC", 'google', start, end)

tickers = ["NYSE:BAC", "NYSE:C", "NYSE:GS", "NYSE:JPM", "NYSE:MS", "NYSE:WFC"]

bank_stocks = pd.concat([BAC, CG, GS, JPMC, MS, WF],axis=1,keys=tickers)

bank_stocks.columns.names = ['Bank Ticker', 'Stock Info']

#print(bank_stocks.head())

#print(bank_stocks.xs(key='Close',axis=1,level='Stock Info').max())

returns = pd.DataFrame()

#print(bank_stocks.xs(key='Close',axis=1,level='Stock Info'))



for banks in bank_stocks.xs(key='Close',axis=1,level='Stock Info'):
    returns[banks + " Return"] = bank_stocks.xs(key='Close',axis=1,level='Stock Info')[banks].pct_change()

#sb.pairplot(returns[1:])
#sb.plt.show()

#print(returns.idxmin())
#print(returns.idxmax())

#print(returns.std())
#print(returns.ix['2015-01-01':'2015-12-31'].std())

#print(returns.ix['2015-01-01':'2015-12-31']['NYSE:MS Return'])

#sb.distplot(returns.ix['2015-01-01':'2015-12-31']['NYSE:MS Return'])
#sb.plt.show()

#sb.distplot(returns.ix['2008-01-01':'2008-12-31']['NYSE:C Return'])
#sb.plt.show()

# Continue here! Not finished yet!

#bank_stocks.xs(key='Close',axis=1,level='Stock Info').plot()
#sb.plt.show()

#bank_stocks.xs(key='Close',axis=1,level='Stock Info')['2008-01-01':'2008-12-31']['NYSE:BAC'].plot()

#sb.tsplot(data=bank_stocks.xs(key='Close',axis=1,level='Stock Info')['2008-01-01':'2008-12-31']['NYSE:BAC'],
# interpolate=True)
#sb.plt.show()

#print(bank_stocks.xs(key='Close',axis=1,level='Stock Info'))


sb.heatmap(data=bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr(), cmap='coolwarm', annot=True)
sb.plt.show()




