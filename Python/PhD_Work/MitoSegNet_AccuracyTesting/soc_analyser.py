"""

15/11/18


"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats.mstats import normaltest, mannwhitneyu, ttest_ind
from Plot_Significance import significance_bar

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)


#path = "C:/Users/Christian/Desktop/Third_CV/Object_Comparison_Data_NoSplitMerge"
path = "C:/Users/Christian/Desktop/Third_CV/Image_sections/sections/Object_Comparison_Data"


file_list = ['MitoSegNet_analysed_data.csv', 'Fiji_U-Net_pretrained_analysed_data.csv', 'Ilastik_analysed_data.csv',
             'Gaussian_analysed_data.csv', 'Hessian_analysed_data.csv', 'Laplacian_analysed_data.csv']

h = []
u = []


all_data = pd.DataFrame(columns=["MitoSegNet", "Pretrained\nFiji U-Net", "Ilastik", "Gaussian", "Hessian", "Laplacian"])


for file, method_name in zip(file_list, all_data):

    if ".csv" in file:

        sheet =  pd.read_csv(path + "/" + file)

        # removing first column
        sheet.drop(sheet.columns[[0, 1]], axis=1, inplace=True)


        #print(sheet)
        #print(np.average([sheet.mean()["area"], sheet.mean()["aspect ratio"], sheet.mean()["eccentricity"],
        #                  sheet.mean()["perimeter"], sheet.mean()["solidity"]]))

        l = sheet.values.tolist()



        # flatten list of lists
        flat_list = [item for sublist in l for item in sublist]

        all_data[method_name] = flat_list


#print(all_data)

#significance_bar(pos_y=1.8, pos_x=[0, 2], bar_y=0.05, p=3, y_dist=0.05, distance=0.1)
#significance_bar(pos_y=2.4, pos_x=[0, 5], bar_y=0.05, p=1, y_dist=0.05, distance=0.1)
#significance_bar(pos_y=2.3, pos_x=[0, 3], bar_y=0.05, p=3, y_dist=0.05, distance=0.1)
#significance_bar(pos_y=2.15, pos_x=[4, 6], bar_y=0.05, p=3, y_dist=0.05, distance=0.1)


#n = sb.boxplot(data=all_data, color="white", fliersize=0)
#n = sb.violinplot(data=all_data, color="white", inner=None)
n = sb.swarmplot(data=all_data, color="black", size=3)
sb.boxplot(data=all_data, color="white", fliersize=0)


n.set_ylabel("Average fold deviation", fontsize=32)
#n.tick_params(labelsize=12)

n.tick_params(axis="x", labelsize=28, rotation=45)
n.tick_params(axis="y", labelsize=28)

"""
all_data["Gaussian"] = np.log(all_data["Gaussian"])
all_data["Hessian"] = np.log(all_data["Hessian"])
all_data["Laplacian"] = np.log(all_data["Laplacian"])
all_data["Ilastik"] = np.log(all_data["Ilastik"])
all_data["MitoSegNet"] = np.log(all_data["MitoSegNet"])


sb.distplot(all_data["Gaussian"], color="blue", label="Gaussian", hist=False)
sb.distplot(all_data["Hessian"], color="orange", label="Hessian", hist=False)
sb.distplot(all_data["Laplacian"], color="green", label="Laplacian", hist=False)
sb.distplot(all_data["Ilastik"], color="red", label="Ilastik", hist=False)
sb.distplot(all_data["MitoSegNet"], color="purple", label="MitoSegNet", hist=False).set(xlabel="Log average fold deviation from gt measurement per object")
"""



print(np.average(all_data["Gaussian"]))
print(np.average(all_data["Hessian"]))
print(np.average(all_data["Laplacian"]))
print(np.average(all_data["Ilastik"]))
print(np.average(all_data["MitoSegNet"]))
print(np.average(all_data["Pretrained\nFiji U-Net"]))

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

#print(ttest_ind(h, u)[1])
print(mannwhitneyu(all_data["Gaussian"], all_data["MitoSegNet"])[1])
print(mannwhitneyu(all_data["Hessian"], all_data["MitoSegNet"])[1])
print(mannwhitneyu(all_data["Laplacian"], all_data["MitoSegNet"])[1])
print(mannwhitneyu(all_data["Ilastik"], all_data["MitoSegNet"])[1])
print(mannwhitneyu(all_data["Pretrained\nFiji U-Net"], all_data["MitoSegNet"])[1])
#print(mannwhitneyu(all_data["Fiji U-Net"], all_data["MitoSegNet"])[1])


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

plt.show()

#sb.distplot(h, color="red")
#sb.distplot(u, color="blue")

