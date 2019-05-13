"""

10/01/2019

intersection over union

common value to describe accuracy of segmentation algorithm

from the u-net imagej plugin (nature methods) paper:

IoU is a widely used measure, for example, in the Pascal VOC challenge16 or
the ISBI cell-tracking challenge17. MIoU∈ [0,1], with 0 meaning no overlap and 1
meaning a perfect match. In our experiments, a value of ∼ 0.7 indicates a good
segmentation result, and a value of ∼ 0.9 is close to human annotation accuracy.

"""

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
from scipy.stats.mstats import normaltest, mannwhitneyu, ttest_ind
from Plot_Significance import significance_bar

import warnings
warnings.simplefilter("ignore", UserWarning)

path = "C:/Users/Christian/Documents/Work/Project_Celegans_Mitochondria_Segmentation/Unet_Segmentation/quantiative_" \
       "segmentation_comparison/Cross_Validation/Third_CV/Complete_images"


gt_folder = "Ground_Truth"
gauss_folder = "Gaussian"
hess_folder = "Hessian"
laplac_folder = "Laplacian"
il_folder = "Ilastik"


unet_folder5 = "U-Net/5_epochs"

gt_directory = path + "/" + gt_folder
gauss_directory = path + "/" + gauss_folder
hess_dir = path + "/" + hess_folder
laplac_dir = path + "/" + laplac_folder
il_dir = path + "/" + il_folder

unet_dir5 = path + "/" + unet_folder5

def iou(pred, true):

    ar_overlap = (np.sum(pred[true == 255]) / 255)
    ar_true = np.sum(true) / 255
    ar_unet1 = np.sum(pred) / 255

    iou = ar_overlap / (ar_true + ar_unet1 - ar_overlap)

    #print(iou)
    return iou

ga_l = []
h_l = []
la_l = []
il_l = []
u_l_1 = []

all_data = pd.DataFrame(columns=["Gaussian", "Hessian", "Laplacian", "Ilastik", "MitoSegNet"]) #, "unet5", "unet10"])


for gt, gauss, hess, laplac, il, unet1 in zip(os.listdir(gt_directory), os.listdir(gauss_directory),
                                                             os.listdir(hess_dir), os.listdir(laplac_dir),
                                                             os.listdir(il_dir), os.listdir(unet_dir5)):


    #print(gt)

    true = cv2.imread(gt_directory + "/" + gt, cv2.IMREAD_GRAYSCALE)

    pred_gauss = cv2.imread(gauss_directory + "/" + gauss, cv2.IMREAD_GRAYSCALE)
    pred_hess = cv2.imread(hess_dir + "/" + hess, cv2.IMREAD_GRAYSCALE)
    pred_laplac = cv2.imread(laplac_dir + "/" + laplac, cv2.IMREAD_GRAYSCALE)
    pred_il = cv2.imread(il_dir + "/" + il, cv2.IMREAD_GRAYSCALE)

    pred_unet1 = cv2.imread(unet_dir5 + "/" + unet1, cv2.IMREAD_GRAYSCALE)


    i_u_g = iou(pred_gauss, true)
    i_u_h = iou(pred_hess, true)
    i_u_l = iou(pred_laplac, true)
    i_u_il = iou(pred_il, true)
    i_u_m = iou(pred_unet1, true)

    ga_l.append(i_u_g)
    h_l.append(i_u_h)
    la_l.append(i_u_l)
    il_l.append(i_u_il)
    u_l_1.append(i_u_m)



all_data["Gaussian"] = ga_l
all_data["Hessian"] = h_l
all_data["Laplacian"] = la_l
all_data["Ilastik"] = il_l
all_data["MitoSegNet"] = u_l_1


significance_bar(pos_y=1, pos_x=[0, 4], bar_y=0.03, p=1, y_dist=0.02, distance=0.1)


sb.boxplot(data=all_data, color="skyblue") #.set(ylabel="Dice coefficient")

plt.ylabel("Intersection over Union", size=18)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)


print(normaltest(all_data["Gaussian"])[1])
print(normaltest(all_data["Hessian"])[1])
print(normaltest(all_data["Laplacian"])[1])
print(normaltest(all_data["Ilastik"])[1])
print(normaltest(all_data["MitoSegNet"])[1])



print("\n")

"""
print(mannwhitneyu(all_data["Gaussian"], all_data["MitoSegNet"])[1])
print(mannwhitneyu(all_data["Hessian"], all_data["MitoSegNet"])[1])
print(mannwhitneyu(all_data["Laplacian"], all_data["MitoSegNet"])[1])
print(mannwhitneyu(all_data["Ilastik"], all_data["MitoSegNet"])[1])
"""

print(ttest_ind(all_data["Gaussian"], all_data["MitoSegNet"])[1])
print(ttest_ind(all_data["Hessian"], all_data["MitoSegNet"])[1])
print(ttest_ind(all_data["Laplacian"], all_data["MitoSegNet"])[1])
print(ttest_ind(all_data["Ilastik"], all_data["MitoSegNet"])[1])



plt.show()



