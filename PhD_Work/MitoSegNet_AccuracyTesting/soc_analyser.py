"""

15/11/18


"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats import normaltest, mannwhitneyu, ttest_ind, f_oneway, kruskal, levene
from scikit_posthocs import posthoc_dunn
from Plot_Significance import significance_bar

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)


path = "C:/Users/Christian/Desktop/Fourth_CV/Object_Comparison_Data_NoSplitMerge"
#path = "C:/Users/Christian/Desktop/Third_CV/Image_sections/sections/Object_Comparison_Data"


file_list = ['MitoSegNet_analysed_data.csv', 'Fiji_U-Net_analysed_data.csv', 'Ilastik_analysed_data.csv',
             'Gaussian_analysed_data.csv', 'Hessian_analysed_data.csv', 'Laplacian_analysed_data.csv']


h = []
u = []


all_data = pd.DataFrame(columns=["MitoSegNet", "Finetuned\nFiji U-Net", "Ilastik", "Gaussian", "Hessian", "Laplacian"])

#all_data = pd.DataFrame(columns=["MitoSegNet", "Pretrained\nFiji U-Net", "Hessian"])

for file, method_name in zip(file_list, all_data):

    if ".csv" in file:

        sheet =  pd.read_csv(path + "/" + file)

        # removing first column
        sheet.drop(sheet.columns[[0]], axis=1, inplace=True)

        #print(np.average([sheet.mean()["area"], sheet.mean()["aspect ratio"], sheet.mean()["eccentricity"],
        #                  sheet.mean()["perimeter"], sheet.mean()["solidity"]]))

        l = sheet.values.tolist()
        #l = sheet["area"].values.tolist()

        #print(l)

        # flatten list of lists
        flat_list = [item for sublist in l for item in sublist]

        all_data[method_name] = flat_list
        #all_data[method_name] = l

"""
msn, ffu, il, gauss, hess, laplac
"""

#significance_bar(pos_y=1.3, pos_x=[0, 2], bar_y=0.03, p=1, y_dist=0.02, distance=0.1)
#significance_bar(pos_y=1.4, pos_x=[0, 3], bar_y=0.03, p=1, y_dist=0.02, distance=0.1)
#significance_bar(pos_y=1.5, pos_x=[0, 4], bar_y=0.03, p=1, y_dist=0.02, distance=0.1)
#significance_bar(pos_y=1.6, pos_x=[0, 5], bar_y=0.03, p=2, y_dist=0.02, distance=0.1)

"""
print(cohens_d(all_data["Gaussian"], all_data["MitoSegNet"]))
print(cohens_d(all_data["Hessian"], all_data["MitoSegNet"]))
print(cohens_d(all_data["Laplacian"], all_data["MitoSegNet"]))
print(cohens_d(all_data["Ilastik"], all_data["MitoSegNet"]))
print(cohens_d(all_data["Pretrained\nFiji U-Net"], all_data["MitoSegNet"]))
"""

# tests null hypothesis that all input samples are from populations with equal variance

print(kruskal(all_data["Gaussian"].tolist(), all_data["Hessian"].tolist(), all_data["Laplacian"].tolist(),
              all_data["Ilastik"].tolist(), all_data["MitoSegNet"].tolist(), all_data["Finetuned\nFiji U-Net"].tolist()))

dt = posthoc_dunn([all_data["Gaussian"].tolist(), all_data["Hessian"].tolist(), all_data["Laplacian"].tolist(),
              all_data["Ilastik"].tolist(), all_data["MitoSegNet"].tolist(), all_data["Finetuned\nFiji U-Net"].tolist()])

dt.to_excel("soc_posthoc.xlsx")


#n = sb.boxplot(data=all_data, color="white", fliersize=0)
#n = sb.violinplot(data=all_data, color="white", inner=None)
n = sb.swarmplot(data=all_data, color="black", size=5)
sb.boxplot(data=all_data, color="white", fliersize=0)


n.set_ylabel("Average fold deviation", fontsize=32)
#n.tick_params(labelsize=12)

#n.tick_params(axis="x", labelsize=28, rotation=45)
n.tick_params(axis="x", labelsize=34, rotation=45)
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


print(np.std(all_data["Gaussian"]))
print(np.std(all_data["Hessian"]))
print(np.std(all_data["Laplacian"]))
print(np.std(all_data["Ilastik"]))
print(np.std(all_data["MitoSegNet"]))
print(np.std(all_data["Finetuned\nFiji U-Net"]))

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

#print(ttest_ind(h, u)[1])
print(mannwhitneyu(all_data["Gaussian"], all_data["MitoSegNet"])[1])
print(mannwhitneyu(all_data["Hessian"], all_data["MitoSegNet"])[1])
print(mannwhitneyu(all_data["Laplacian"], all_data["MitoSegNet"])[1])
print(mannwhitneyu(all_data["Ilastik"], all_data["MitoSegNet"])[1])
print(mannwhitneyu(all_data["Finetuned\nFiji U-Net"], all_data["MitoSegNet"])[1])
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
print(cohens_d(all_data["Finetuned\nFiji U-Net"], all_data["MitoSegNet"]))
#print(cohens_d(all_data["Fiji U-Net"], all_data["MitoSegNet"]))

plt.show()

#sb.distplot(h, color="red")
#sb.distplot(u, color="blue")

