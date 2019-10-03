import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sb


pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)


# enter the path under which you find the file you want to read
path = "C:/Users/Christian/Desktop"

# enter the name of the file, including the file extension (like .csv or .xlsx)
filename = "Raw_Data_Organized_Triplicate.xlsx"

# select the sheet you want to read in
#sheet = "Fluor. C3 Raw Data All Tripl"
sheet = "Fluor. C2 Raw Data All Trip"

table = pd.read_excel(path + os.sep + filename, sheet_name=sheet)

# rows / columns
table1 = table.iloc[1:7, 1:20]

#print(table1)


names_list = table.iloc[1:7, 0].tolist()


final_table = table1.T

final_table.columns = names_list

new_table = final_table.drop(['KO DMSO', 'Par. DMSO'], axis=1)

#print(new_table)

ko_av = np.average(final_table['KO DMSO'].tolist())
par_av = np.average(final_table['Par. DMSO'].tolist())

norm_kozm = new_table['KO ZM'].div(ko_av)
norm_kosw = new_table['KO S+W'].div(ko_av)

norm_parzm = new_table['Par. ZM'].div(par_av)
norm_parsw = new_table['Par. S+W'].div(par_av)


new_fin_table = pd.concat([norm_kozm, norm_parzm, norm_kosw, norm_parsw], axis=1, sort=False)

#new_fin_table["Grouping"] = [1]*6+[2]*6+[3]*6

#testdata = sb.load_dataset("tips")
#print(testdata)

print(new_fin_table)

#new_fin_table.to_excel("Normalized_values.xlsx")

"""
n = sb.boxplot(data=new_fin_table, fliersize=0)
sb.swarmplot(data=new_fin_table, color="black", size=10)

n.set_ylabel("Fluorescence intensity", fontsize=32)
n.tick_params(axis="x", labelsize=34, rotation=45)
n.tick_params(axis="y", labelsize=28)
"""


sb.distplot(new_fin_table["KO ZM"].tolist())

plt.show()
