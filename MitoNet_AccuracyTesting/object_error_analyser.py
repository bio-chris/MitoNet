"""

28/11/18

analyser script for the Error_Counting_Data

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


path = "C:/Users/Christian/Documents/Work/Project_Celegans_Mitochondria_Segmentation/Unet_Segmentation/quantiative_" \
       "segmentation_comparison/Cross_Validation/Third_CV/Error_Counting_Data"


file_list = os.listdir(path)
# switching element index in list
file_list[2], file_list[3] = file_list[3], file_list[2]

all_data = pd.DataFrame(columns=["Gaussian", "Hessian", "Laplacian", "Ilastik", "MitoNet"])

print(file_list)

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

            #print(merge,split,add)
            l.append(merge+split+add)


        all_data[method] = l

print(all_data)



"""
print(normaltest(all_data["Gaussian"])[1])
print(normaltest(all_data["Hessian"])[1])
print(normaltest(all_data["Laplacian"])[1])
print(normaltest(all_data["Ilastik"])[1])
print(normaltest(all_data["MitoNet"])[1], "\n")
"""

method = "Ilastik"

#print(mannwhitneyu(all_data[method], all_data["MitoNet"])[1])
#print(ttest_ind(all_data[method], all_data["MitoNet"])[1])

"""
all_data["Gaussian"] = np.log(all_data["Gaussian"])
all_data["Hessian"] = np.log(all_data["Hessian"])
all_data["Laplacian"] = np.log(all_data["Laplacian"])
all_data["Ilastik"] = np.log(all_data["Ilastik"])
all_data["MitoNet"] = np.log(all_data["MitoNet"])
"""
"""
sb.distplot(all_data["Gaussian"], color="blue", label="Gaussian", hist=False)
sb.distplot(all_data["Hessian"], color="orange", label="Hessian", hist=False)
sb.distplot(all_data["Laplacian"], color="green", label="Laplacian", hist=False)
sb.distplot(all_data["Ilastik"], color="red", label="Ilastik", hist=False)
sb.distplot(all_data["MitoNet"], color="purple", label="MitoNet", hist=False).set(xlabel="Percent of wrongly segmented objects")
"""

sb.boxplot(data=all_data).set(ylabel="Percent of wrongly segmented objects")

significance_bar(pos_y=1.4, pos_x=[0, 4], bar_y=0.03, p=2, y_dist=0.1, distance=0.1)
significance_bar(pos_y=1.1, pos_x=[2, 4], bar_y=0.03, p=3, y_dist=0.1, distance=0.1)

plt.show()