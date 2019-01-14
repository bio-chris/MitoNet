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


path = "C:/Users/Christian/Documents/Work/Project_Celegans_Mitochondria_Segmentation/Unet_Segmentation/quantiative_" \
       "segmentation_comparison/Cross_Validation/Third_CV/Object_Comparison_Data_Summarized"

file_list = os.listdir(path)

h = []
u = []

all_data = pd.DataFrame(columns=["Gaussian", "Hessian", "Laplacian", "Ilastik", "MitoNet"])

file_list[2], file_list[3] = file_list[3], file_list[2]

for file, method_name in zip(file_list, all_data):

    if ".csv" in file:

        #print(file)

        sheet =  pd.read_csv(path + "/" + file)

        # removing first column
        sheet.drop(sheet.columns[[0, 1]], axis=1, inplace=True)


        #print(np.average([sheet.mean()["area"], sheet.mean()["aspect ratio"], sheet.mean()["eccentricity"],
        #                  sheet.mean()["perimeter"], sheet.mean()["solidity"]]))

        l = sheet.values.tolist()

        # flatten list of lists
        flat_list = [item for sublist in l for item in sublist]

        all_data[method_name] = flat_list

        """
        if "Ilastik" in file:
    
            for index, row in sheet.iterrows():
                h.append(np.average(row))
    
        if "U-Net10" in file:
    
            for index, row in sheet.iterrows():
                u.append(np.average(row))
        """

#print(all_data)

significance_bar(pos_y=4, pos_x=[0, 4], bar_y=0.03, p=3, y_dist=0.1, distance=0.1)
significance_bar(pos_y=3, pos_x=[2, 4], bar_y=0.03, p=1, y_dist=0.1, distance=0.1)


sb.boxplot(data=all_data, fliersize=0).set(ylabel="Average fold deviation from gt measurement")

"""
all_data["Gaussian"] = np.log(all_data["Gaussian"])
all_data["Hessian"] = np.log(all_data["Hessian"])
all_data["Laplacian"] = np.log(all_data["Laplacian"])
all_data["Ilastik"] = np.log(all_data["Ilastik"])
all_data["MitoNet"] = np.log(all_data["MitoNet"])


sb.distplot(all_data["Gaussian"], color="blue", label="Gaussian", hist=False)
sb.distplot(all_data["Hessian"], color="orange", label="Hessian", hist=False)
sb.distplot(all_data["Laplacian"], color="green", label="Laplacian", hist=False)
sb.distplot(all_data["Ilastik"], color="red", label="Ilastik", hist=False)
sb.distplot(all_data["MitoNet"], color="purple", label="MitoNet", hist=False).set(xlabel="Log average fold deviation from gt measurement per object")
"""

plt.show()




"""
print(normaltest(all_data["Gaussian"])[1])
print(normaltest(all_data["Hessian"])[1])
print(normaltest(all_data["Laplacian"])[1])
print(normaltest(all_data["Ilastik"])[1])
print(normaltest(all_data["MitoNet"])[1])
"""

#print(ttest_ind(h, u)[1])
print(mannwhitneyu(all_data["Ilastik"], all_data["MitoNet"])[1])

#sb.distplot(h, color="red")
#sb.distplot(u, color="blue")

