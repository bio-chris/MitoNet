"""

28/11/18

analyser script for the Error_Counting_Data

"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats import normaltest, mannwhitneyu, ttest_ind, f_oneway, levene
from scikit_posthocs import posthoc_tukey
from Plot_Significance import significance_bar

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)


pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)

path = "C:/Users/Christian/Desktop/Fourth_CV/Error_Counting_Data"


file_list = ['MitoSegNet_error_counting_data.csv', 'Fiji_U-Net_error_counting_data.csv', 'Ilastik_error_counting_data.csv',
             'Gaussian_error_counting_data.csv', 'Hessian_error_counting_data.csv', 'Laplacian_error_counting_data.csv']

#file_list = ['MitoSegNet_error_counting_data.csv', 'Fiji_U-Net_pretrained_error_counting_data.csv',  'Hessian_error_counting_data.csv']

all_data = pd.DataFrame(columns=["MitoSegNet", "Finetuned\nFiji U-Net", "Ilastik", "Gaussian", "Hessian", "Laplacian"])

#all_data = pd.DataFrame(columns=["MitoSegNet", "Pretrained\nFiji U-Net", "Hessian"])


for file, method in zip(file_list, all_data):

    if ".csv" in file:

        #print(file)

        table = pd.read_csv(path + "/" + file)

        # removing first column
        table.drop(table.columns[[0, 1]], axis=1, inplace=True)

        """
        l = []
        
        for gt_obj, seg_obj in zip(table["nr of gt objects"], table["nr of seg objects"]):

            if gt_obj > seg_obj:

                dev = gt_obj/seg_obj

            else:

                dev = seg_obj/gt_obj


            l.append(dev)
        """


        l = []
        for merge, split, add in zip(table["falsely merged"], table["falsely split"], table["falsely added"]):
        #for missing in table["missing"]:

            #print(merge,split,add)

            l.append(merge+split+add)
            #l.append(missing)


        all_data[method] = l


#print(all_data)

print(levene(all_data["Gaussian"].tolist(), all_data["Hessian"].tolist(), all_data["Laplacian"].tolist(),
              all_data["Ilastik"].tolist(), all_data["MitoSegNet"].tolist(), all_data["Finetuned\nFiji U-Net"].tolist()))

print(f_oneway(all_data["Gaussian"].tolist(), all_data["Hessian"].tolist(), all_data["Laplacian"].tolist(),
              all_data["Ilastik"].tolist(), all_data["MitoSegNet"].tolist(), all_data["Finetuned\nFiji U-Net"].tolist()))

x = [all_data["Gaussian"].tolist(), all_data["Hessian"].tolist(), all_data["Laplacian"].tolist(),
     all_data["Ilastik"].tolist(), all_data["MitoSegNet"].tolist(), all_data["Finetuned\nFiji U-Net"].tolist()]

#print(posthoc_tukey(x))

x = all_data
x = x.melt(var_name='groups', value_name='values')
print(posthoc_tukey(x, val_col='values', group_col='groups'))



print("\n")

print(np.average(all_data["Gaussian"]))
print(np.average(all_data["Hessian"]))
print(np.average(all_data["Laplacian"]))
print(np.average(all_data["Ilastik"]))
print(np.average(all_data["MitoSegNet"]))
print(np.average(all_data["Finetuned\nFiji U-Net"]))

print("\n")


#print(kruskal(all_data["Gaussian"].tolist(), all_data["Hessian"].tolist(), all_data["Laplacian"].tolist(),
#              all_data["Ilastik"].tolist(), all_data["MitoSegNet"].tolist(), all_data["Finetuned\nFiji U-Net"].tolist()))


#print(posthoc_dunn([all_data["Gaussian"].tolist(), all_data["Hessian"].tolist(), all_data["Laplacian"].tolist(),
#              all_data["Ilastik"].tolist(), all_data["MitoSegNet"].tolist(), all_data["Finetuned\nFiji U-Net"].tolist()]))


#"""
print(normaltest(all_data["Gaussian"])[1])
print(normaltest(all_data["Hessian"])[1])
print(normaltest(all_data["Laplacian"])[1])
print(normaltest(all_data["Ilastik"])[1])
print(normaltest(all_data["MitoSegNet"])[1])
print(normaltest(all_data["Finetuned\nFiji U-Net"])[1])
#print(normaltest(all_data["Fiji U-Net"])[1], "\n")
#"""

print("\n")

#method = "Ilastik"

#print(mannwhitneyu(all_data[method], all_data["MitoSegNet"])[1])
#print(ttest_ind(all_data[method], all_data["MitoSegNet"])[1])

print(mannwhitneyu(all_data["Gaussian"], all_data["MitoSegNet"])[1])
print(mannwhitneyu(all_data["Hessian"], all_data["MitoSegNet"])[1])
print(mannwhitneyu(all_data["Laplacian"], all_data["MitoSegNet"])[1])
print(mannwhitneyu(all_data["Ilastik"], all_data["MitoSegNet"])[1])
print(mannwhitneyu(all_data["Finetuned\nFiji U-Net"], all_data["MitoSegNet"])[1])
#print(ttest_ind(all_data["Fiji U-Net"], all_data["MitoSegNet"])[1])

print("\n")

"""
all_data["Gaussian"] = np.log(all_data["Gaussian"])
all_data["Hessian"] = np.log(all_data["Hessian"])
all_data["Laplacian"] = np.log(all_data["Laplacian"])
all_data["Ilastik"] = np.log(all_data["Ilastik"])
all_data["MitoSegNet"] = np.log(all_data["MitoSegNet"])
"""
"""
sb.distplot(all_data["Gaussian"], color="blue", label="Gaussian", hist=False)
sb.distplot(all_data["Hessian"], color="orange", label="Hessian", hist=False)
sb.distplot(all_data["Laplacian"], color="green", label="Laplacian", hist=False)
sb.distplot(all_data["Ilastik"], color="red", label="Ilastik", hist=False)
sb.distplot(all_data["MitoSegNet"], color="purple", label="MitoSegNet", hist=False).set(xlabel="Percent of wrongly segmented objects")
"""


def cohens_d(data1, data2):

    p_std = np.sqrt(((len(data1)-1)*np.var(data1)+(len(data2)-1)*np.var(data2))/(len(data1)+len(data2)-2))

    cohens_d = np.abs(np.average(data1) - np.average(data2)) / p_std

    return cohens_d


print(cohens_d(all_data["Gaussian"], all_data["MitoSegNet"]))
print(cohens_d(all_data["Hessian"], all_data["MitoSegNet"]))
print(cohens_d(all_data["Laplacian"], all_data["MitoSegNet"]))
print(cohens_d(all_data["Ilastik"], all_data["MitoSegNet"]))
print(cohens_d(all_data["Finetuned\nFiji U-Net"], all_data["MitoSegNet"]))
#print(cohens_d(all_data["Fiji U-Net"], all_data["MitoSegNet"]))



#n = sb.violinplot(data=all_data, color="white", inner=None)#.set(ylabel="Percent of wrongly\nsegmented objects")
n = sb.swarmplot(data=all_data, color="black")
sb.boxplot(data=all_data, color="white", fliersize=0)


n.set_ylabel("Missing objects", fontsize=32)

#n.set_ylabel("Percent of missing objects", fontsize=18)

#n.tick_params(labelsize=14)

n.tick_params(axis="x", labelsize=34, rotation=45)
n.tick_params(axis="y", labelsize=28)

#significance_bar(pos_y=1.3, pos_x=[0, 5], bar_y=0.03, p=2, y_dist=0.05, distance=0.1)
#significance_bar(pos_y=1.2, pos_x=[0, 3], bar_y=0.03, p=2, y_dist=0.05, distance=0.1)


plt.show()