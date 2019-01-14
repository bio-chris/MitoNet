import numpy as np
import pandas as pd

"""
file = pd.read_csv('Salaries.csv')


#print(file.head())

#print(file.info())

# Average base pay
print(file['BasePay'].mean())

# Max Overtime pay
print(file['OvertimePay'].max())


print(file.loc[file['EmployeeName']=='JOSEPH DRISCOLL']['JobTitle'])
# Solution from Instructor
#print(file[file['EmployeeName'] == 'JOSEPH DRISCOLL'])


print(file.loc[file['EmployeeName']=='JOSEPH DRISCOLL']['TotalPayBenefits'])


print(file.loc[file['TotalPayBenefits'].idxmax()]['EmployeeName'])


print(file.loc[file['TotalPayBenefits'].idxmin()]['EmployeeName'])

byComp = file.groupby('Year')

print(byComp['BasePay'].mean())

print(file['JobTitle'].nunique())


print(file['JobTitle'].value_counts().nlargest(5))


print(len(file[file['Year']==2013]['JobTitle'].unique()))


# Solution from Instructor

print(sum(file[file['Year']==2013]['JobTitle'].value_counts() == 1))

def contains_chief(title):
    if "chief" in title.lower():
        return True
    else:
        return False

print(sum(file['JobTitle'].apply(lambda x: contains_chief(x))))

#print(file['JobTitle'].str.contains("Chief"))

print(len(file[file['JobTitle'].str.contains("Chief")]['JobTitle']))


file['title_len'] = file['JobTitle'].apply(len)

print(file[['TotalPayBenefits', 'title_len']].corr())

"""
file = pd.read_csv('Ecommerce Purchases.csv')

#print(file.head())

#print(file.info())

#print(file["Purchase Price"].mean())

#print(file["Purchase Price"].max())
#print(file["Purchase Price"].min())


#print(len(file[file["Language"] == "en"]))

#print(len(file[file["Job"] == "Lawyer"]))

#print(len(file[file["AM or PM"] == "PM"]))
#print(len(file[file["AM or PM"] == "AM"]))

#print(file["Job"].value_counts().nlargest(5))

#print(file[file["Lot"] == "90 WT"]["Purchase Price"])

#print(file[file["Credit Card"] == 4926535242672853]["Email"])

#print(len(file[(file["CC Provider"] == "American Express") & (file["Purchase Price"] > 95)]))

"""
def exp_date(date):
    list = date.split("/")
    if int((list[1])) == 25:
        return True
    else:
        return False

print(len(file[file["CC Exp Date"].apply(lambda x: exp_date(x))]))


def mail(provider):
    list = provider.split("@")
    return (list[1])

print(file["Email"].apply(mail).value_counts().nlargest(5))
"""


