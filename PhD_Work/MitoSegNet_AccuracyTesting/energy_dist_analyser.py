"""

12/12/18

Create a python generated stacked bar figure of the morphological accuracy

"""

import os
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.mstats import normaltest, mannwhitneyu, ttest_ind, kruskal
from scikit_posthocs import posthoc_dunn
from Plot_Significance import significance_bar

import warnings
warnings.simplefilter("ignore", UserWarning)

path = "C:/Users/Christian/Desktop/Fourth_CV/Morph_Energy_Distance_Extended"
#path = "C:/Users/Christian/Desktop/Fourth_CV/Morph_Distribution_Comparison_with_cohensd"




#file_list = ['MitoSegNet_EffectSize.csv', 'Fiji_U-Net_EffectSize.csv', 'Ilastik_EffectSize.csv',
#             'Gaussian_EffectSize.csv', 'Hessian_EffectSize.csv', 'Laplacian_EffectSize.csv']


file_list = ['MitoSegNet_EnergyDistance.csv', 'Fiji_U-Net_EnergyDistance.csv', 'Ilastik_EnergyDistance.csv',
             'Gaussian_EnergyDistance.csv', 'Hessian_EnergyDistance.csv', 'Laplacian_EnergyDistance.csv']

all_data = pd.DataFrame(columns=["MitoSegNet", "Finetuned\nFiji U-Net", "Ilastik", "Gaussian", "Hessian", "Laplacian"])


#descriptor = "Mean intensity"

for file, method in zip(file_list, all_data):

    if ".csv" in file:

        print(file, method)

        table = pd.read_csv(path + "/" + file)

        # removing first column
        table.drop(table.columns[[0,1]], axis=1, inplace=True)

        #print(table)

        l = table.values.tolist()
        #l = table[descriptor].tolist()

        flat_list = [item for sublist in l for item in sublist]

        all_data[method] = flat_list #l
        #all_data[method] = l

#print(len(all_data["Gaussian"]))

#print(all_data)

print("\n")

print(np.median(all_data["Gaussian"]))
print(np.median(all_data["Hessian"]))
print(np.median(all_data["Laplacian"]))
print(np.median(all_data["Ilastik"]))
print(np.median(all_data["MitoSegNet"]))
print(np.median(all_data["Finetuned\nFiji U-Net"]))

print("\n")

#"""
print(normaltest(all_data["Gaussian"])[1])
print(normaltest(all_data["Hessian"])[1])
print(normaltest(all_data["Laplacian"])[1])
print(normaltest(all_data["Ilastik"])[1])
print(normaltest(all_data["MitoSegNet"])[1])
print(normaltest(all_data["Finetuned\nFiji U-Net"])[1])
#print(normaltest(all_data["Fiji U-Net"])[1])
#"""

print("\n")

print(mannwhitneyu(all_data["Gaussian"], all_data["MitoSegNet"])[1])
print(mannwhitneyu(all_data["Hessian"], all_data["MitoSegNet"])[1])
print(mannwhitneyu(all_data["Laplacian"], all_data["MitoSegNet"])[1])
print(mannwhitneyu(all_data["Ilastik"], all_data["MitoSegNet"])[1])
print(mannwhitneyu(all_data["Finetuned\nFiji U-Net"], all_data["MitoSegNet"])[1])
#print(mannwhitneyu(all_data["Fiji U-Net"], all_data["MitoSegNet"])[1])

p_g = mannwhitneyu(all_data["Gaussian"], all_data["MitoSegNet"])[1]
p_h = mannwhitneyu(all_data["Hessian"], all_data["MitoSegNet"])[1]
p_l = mannwhitneyu(all_data["Laplacian"], all_data["MitoSegNet"])[1]
p_i = mannwhitneyu(all_data["Ilastik"], all_data["MitoSegNet"])[1]
p_f = mannwhitneyu(all_data["Finetuned\nFiji U-Net"], all_data["MitoSegNet"])[1]

def star_counter(value):

    if value > 0.05:
        p = 0

    elif 0.01 < value < 0.05:
        p = 1

    elif 0.001 < value < 0.01:
        p = 2

    else:
        p = 3

    return p

print("\n")

print(mannwhitneyu(all_data["Gaussian"], all_data["Finetuned\nFiji U-Net"])[1])
print(mannwhitneyu(all_data["Hessian"], all_data["Finetuned\nFiji U-Net"])[1])
print(mannwhitneyu(all_data["Laplacian"], all_data["Finetuned\nFiji U-Net"])[1])
print(mannwhitneyu(all_data["Ilastik"], all_data["Finetuned\nFiji U-Net"])[1])

#print(np.median(all_data["Hessian"]), np.median(all_data["Ilastik"]), np.median(all_data["MitoSegNet"]))
#print(np.average(all_data["Hessian"]), np.average(all_data["Ilastik"]), np.average(all_data["MitoSegNet"]))

print(kruskal(all_data["Gaussian"].tolist(), all_data["Hessian"].tolist(), all_data["Laplacian"].tolist(),
              all_data["Ilastik"].tolist(), all_data["MitoSegNet"].tolist(), all_data["Finetuned\nFiji U-Net"].tolist()))


dt = posthoc_dunn([all_data["Gaussian"].tolist(), all_data["Hessian"].tolist(), all_data["Laplacian"].tolist(),
              all_data["Ilastik"].tolist(), all_data["MitoSegNet"].tolist(), all_data["Finetuned\nFiji U-Net"].tolist()])


dt.to_excel("ed_posthoc.xlsx")


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
print(cohens_d(all_data["Finetuned\nFiji U-Net"], all_data["MitoSegNet"]))
#print(cohens_d(all_data["Fiji U-Net"], all_data["MitoSegNet"]))


pos_y_start = 11
dist_bar_y = 0.2
if p_f < 0.05:
    significance_bar(pos_y=pos_y_start, pos_x=[0, 1], bar_y=dist_bar_y, p=star_counter(p_f), y_dist=dist_bar_y, distance=0.11)
if p_i < 0.05:
    significance_bar(pos_y=pos_y_start+0.5, pos_x=[0, 2], bar_y=dist_bar_y, p=star_counter(p_i), y_dist=dist_bar_y, distance=0.11)
if p_g < 0.05:
    significance_bar(pos_y=pos_y_start+1, pos_x=[0, 3], bar_y=dist_bar_y, p=star_counter(p_g), y_dist=dist_bar_y, distance=0.11)
if p_h < 0.05:
    significance_bar(pos_y=pos_y_start+1.5, pos_x=[0, 4], bar_y=dist_bar_y, p=star_counter(p_h), y_dist=dist_bar_y, distance=0.11)
if p_l < 0.05:
    significance_bar(pos_y=pos_y_start+2, pos_x=[0, 5], bar_y=dist_bar_y, p=star_counter(p_l), y_dist=dist_bar_y, distance=0.11)



#sb.violinplot(data=all_data, color="white", inner=None)
#sb.boxplot(data=all_data, color="white", fliersize=0)

sb.violinplot(data=all_data, color="grey", inner=None)
#sb.swarmplot(data=all_data, color="black", size=6)

plt.ylabel("Energy distance", size=32)
#plt.ylabel("Energy distance\n(" +descriptor+ ")", size=32)
plt.yticks(fontsize=28)
plt.xticks(fontsize=28, rotation=45)



plt.show()

#sb.distplot(all_data["Hessian"], color="red")
#sb.distplot(all_data["MitoNet"], color="blue")
#plt.show()






