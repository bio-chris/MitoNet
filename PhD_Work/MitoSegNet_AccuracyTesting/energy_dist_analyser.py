"""

12/12/18

Create a python generated stacked bar figure of the morphological accuracy

"""

import os
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.mstats import normaltest, mannwhitneyu, ttest_ind
from Plot_Significance import significance_bar

import warnings
warnings.simplefilter("ignore", UserWarning)

path = "C:/Users/Christian/Desktop/Third_CV/Morph_Energy_Distance"





file_list = ['MitoNet_EnergyDistance.csv', 'Fiji_U-Net_pretrained_EnergyDistance.csv', 'Ilastik_EnergyDistance.csv',
             'Gaussian_EnergyDistance.csv', 'Hessian_EnergyDistance.csv', 'Laplacian_EnergyDistance.csv']

all_data = pd.DataFrame(columns=["MitoSegNet", "Pretrained\nFiji U-Net", "Ilastik", "Gaussian", "Hessian", "Laplacian"])


descriptor = "Solidity"
for file, method in zip(file_list, all_data):

    if ".csv" in file:

        print(file, method)

        table = pd.read_csv(path + "/" + file)

        # removing first column
        table.drop(table.columns[[0,1]], axis=1, inplace=True)

        l = table.values.tolist()
        #l = table[descriptor].tolist()

        flat_list = [item for sublist in l for item in sublist]


        all_data[method] = flat_list #l


#print(len(all_data["Gaussian"]))

#print(all_data)

print(np.std(all_data["Gaussian"]))
print(np.std(all_data["Hessian"]))
print(np.std(all_data["Laplacian"]))
print(np.std(all_data["Ilastik"]))
print(np.std(all_data["MitoSegNet"]))
print(np.std(all_data["Pretrained\nFiji U-Net"]))

print("\n")

#"""
print(normaltest(all_data["Gaussian"])[1])
print(normaltest(all_data["Hessian"])[1])
print(normaltest(all_data["Laplacian"])[1])
print(normaltest(all_data["Ilastik"])[1])
print(normaltest(all_data["MitoSegNet"])[1])
print(normaltest(all_data["Pretrained\nFiji U-Net"])[1])
#print(normaltest(all_data["Fiji U-Net"])[1])
#"""

print("\n")

print(mannwhitneyu(all_data["Gaussian"], all_data["MitoSegNet"])[1])
print(mannwhitneyu(all_data["Hessian"], all_data["MitoSegNet"])[1])
print(mannwhitneyu(all_data["Laplacian"], all_data["MitoSegNet"])[1])
print(mannwhitneyu(all_data["Ilastik"], all_data["MitoSegNet"])[1])
print(mannwhitneyu(all_data["Pretrained\nFiji U-Net"], all_data["MitoSegNet"])[1])
#print(mannwhitneyu(all_data["Fiji U-Net"], all_data["MitoSegNet"])[1])

#print(np.median(all_data["Hessian"]), np.median(all_data["Ilastik"]), np.median(all_data["MitoSegNet"]))
#print(np.average(all_data["Hessian"]), np.average(all_data["Ilastik"]), np.average(all_data["MitoSegNet"]))


# pooled standard deviation for calculation of effect size (cohen's d)
def cohens_d(data1, data2):

    p_std = np.sqrt(((len(data1)-1)*np.var(data1)+(len(data2)-1)*np.var(data2))/(len(data1)+len(data2)-2))

    cohens_d = np.abs(np.average(data1) - np.average(data2)) / p_std

    return cohens_d

print("\n")

print(cohens_d(all_data["Gaussian"], all_data["MitoSegNet"]))
print(cohens_d(all_data["Hessian"], all_data["MitoSegNet"]))
print(cohens_d(all_data["Laplacian"], all_data["MitoSegNet"]))
print(cohens_d(all_data["Ilastik"], all_data["MitoSegNet"]))
print(cohens_d(all_data["Pretrained\nFiji U-Net"], all_data["MitoSegNet"]))
#print(cohens_d(all_data["Fiji U-Net"], all_data["MitoSegNet"]))


dist_bar_y = 0.2

significance_bar(pos_y=5.5, pos_x=[0, 3], bar_y=dist_bar_y, p=2, y_dist=dist_bar_y, distance=0.11)
significance_bar(pos_y=9.2, pos_x=[0, 5], bar_y=dist_bar_y, p=2, y_dist=dist_bar_y, distance=0.11)
#significance_bar(pos_y=6.5, pos_x=[4, 6], bar_y=dist_bar_y, p=2, y_dist=dist_bar_y, distance=0.11)
#significance_bar(pos_y=1.3, pos_x=[3, 4], bar_y=dist_bar_y, p=2, y_dist=dist_bar_y, distance=0.11)
#significance_bar(pos_y=1.8, pos_x=[4, 6], bar_y=dist_bar_y, p=2, y_dist=dist_bar_y, distance=0.11)


#sb.violinplot(data=all_data, color="white", inner=None)
sb.boxplot(data=all_data, color="white", fliersize=0)
sb.swarmplot(data=all_data, color="black", size=3)

plt.ylabel("Energy distance\n(All descriptors)", size=18)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)

plt.show()

#sb.distplot(all_data["Hessian"], color="red")
#sb.distplot(all_data["MitoNet"], color="blue")
#plt.show()






