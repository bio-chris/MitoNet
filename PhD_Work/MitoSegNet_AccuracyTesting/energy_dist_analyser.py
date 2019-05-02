"""

12/12/18

Create a python generated stacked bar figure of the morphological accuracy

"""

import os
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.mstats import normaltest, mannwhitneyu
from Plot_Significance import significance_bar


path = "C:/Users/Christian/Documents/Work/Project_Celegans_Mitochondria_Segmentation/Unet_Segmentation/quantiative_" \
       "segmentation_comparison/Cross_Validation/Third_CV/Morph_Energy_Distance"



file_list = os.listdir(path)

# switch list element at position 2 with list element at position 3 (ilastik<>laplacian)
file_list[2], file_list[3] = file_list[3], file_list[2]



all_data = pd.DataFrame(columns=["Gaussian", "Hessian", "Laplacian", "Ilastik", "MitoSegNet"])


for file, method in zip(file_list, all_data):

    if ".csv" in file:

        #print(file, method)

        table = pd.read_csv(path + "/" + file)

        # removing first column
        table.drop(table.columns[[0,1]], axis=1, inplace=True)

        l = table.values.tolist()

        flat_list = [item for sublist in l for item in sublist]


        all_data[method] = flat_list



#"""
print(normaltest(all_data["Gaussian"])[1])
print(normaltest(all_data["Hessian"])[1])
print(normaltest(all_data["Laplacian"])[1])
print(normaltest(all_data["Ilastik"])[1])
print(normaltest(all_data["MitoSegNet"])[1])
#"""

print("\n")

print(mannwhitneyu(all_data["Gaussian"], all_data["MitoSegNet"])[1])
print(mannwhitneyu(all_data["Hessian"], all_data["MitoSegNet"])[1])
print(mannwhitneyu(all_data["Laplacian"], all_data["MitoSegNet"])[1])
print(mannwhitneyu(all_data["Ilastik"], all_data["MitoSegNet"])[1])

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



significance_bar(pos_y=8, pos_x=[0, 4], bar_y=0.2, p=2, y_dist=0.2, distance=0.11)
significance_bar(pos_y=7.2, pos_x=[2, 4], bar_y=0.2, p=3, y_dist=0.2, distance=0.11)


sb.boxplot(data=all_data, color="skyblue", fliersize=0)

plt.ylabel("Energy distance", size=18)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)

plt.show()

#sb.distplot(all_data["Hessian"], color="red")
#sb.distplot(all_data["MitoNet"], color="blue")
#plt.show()






