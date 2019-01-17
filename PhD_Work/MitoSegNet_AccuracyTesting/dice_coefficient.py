"""

15/11/18

Calculating the dice coefficient of segmentation predictions (compared to hand-generated ground truth).

Can also generate boxplots of dice coefficient values for different segmentation approaches


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
warnings.simplefilter("ignore", FutureWarning)


path = "C:/Users/Christian/Documents/Work/Project_Celegans_Mitochondria_Segmentation/Unet_Segmentation/quantiative_" \
       "segmentation_comparison/Cross_Validation/Third_CV/Complete_images"


gt_folder = "Ground_Truth"
gauss_folder = "Gaussian"
hess_folder = "Hessian"
laplac_folder = "Laplacian"
il_folder = "Ilastik/post_processed"

unet_folder1 = "U-Net/1_epoch"
unet_folder5 = "U-Net/5_epochs"
unet_folder10 = "U-Net/10_epochs"


gt_directory = path + "/" + gt_folder
gauss_directory = path + "/" + gauss_folder
hess_dir = path + "/" + hess_folder
laplac_dir = path + "/" + laplac_folder
il_dir = path + "/" + il_folder

unet_dir1 = path + "/" + unet_folder1
unet_dir5 = path + "/" + unet_folder5
unet_dir10 = path + "/" + unet_folder10

ga_l = []
h_l = []
la_l = []
il_l = []

u_l_1 = []
#u_l_5 = []
#u_l_10 = []

all_data = pd.DataFrame(columns=["Gaussian", "Hessian", "Laplacian", "Ilastik", "MitoNet"]) #, "unet5", "unet10"])

for gt, gauss, hess, laplac, il, unet1 in zip(os.listdir(gt_directory), os.listdir(gauss_directory),
                                                             os.listdir(hess_dir), os.listdir(laplac_dir),
                                                             os.listdir(il_dir), os.listdir(unet_dir5)):
                                                             #os.listdir(unet_dir5), os.listdir(unet_dir10)):

    #print(gt)

    true = cv2.imread(gt_directory + "/" + gt, cv2.IMREAD_GRAYSCALE)

    pred_gauss = cv2.imread(gauss_directory + "/" + gauss, cv2.IMREAD_GRAYSCALE)
    pred_hess = cv2.imread(hess_dir + "/" + hess, cv2.IMREAD_GRAYSCALE)
    pred_laplac = cv2.imread(laplac_dir + "/" + laplac, cv2.IMREAD_GRAYSCALE)
    pred_il = cv2.imread(il_dir + "/" + il, cv2.IMREAD_GRAYSCALE)

    pred_unet1 = cv2.imread(unet_dir5 + "/" + unet1, cv2.IMREAD_GRAYSCALE)
    #pred_unet5 = cv2.imread(unet_dir5 + "/" + unet5, cv2.IMREAD_GRAYSCALE)
    #pred_unet10 = cv2.imread(unet_dir10 + "/" + unet10, cv2.IMREAD_GRAYSCALE)


    dice_ga = (np.sum(pred_gauss[true == 255]) * 2.0) / (np.sum(pred_gauss) + np.sum(true))
    dice_h = (np.sum(pred_hess[true == 255]) * 2.0) / (np.sum(pred_hess) + np.sum(true))
    dice_l = (np.sum(pred_laplac[true == 255]) * 2.0) / (np.sum(pred_laplac) + np.sum(true))
    dice_il = (np.sum(pred_il[true == 255]) * 2.0) / (np.sum(pred_il) + np.sum(true))

    dice_u1 = (np.sum(pred_unet1[true == 255]) * 2.0) / (np.sum(pred_unet1) + np.sum(true))
    #dice_u5 = (np.sum(pred_unet5[true == 255]) * 2.0) / (np.sum(pred_unet5) + np.sum(true))
    #dice_u10 = (np.sum(pred_unet10[true == 255]) * 2.0) / (np.sum(pred_unet10) + np.sum(true))

    #print(dice_u)

    ga_l.append(dice_ga)
    h_l.append(dice_h)
    la_l.append(dice_l)
    il_l.append(dice_il)

    u_l_1.append(dice_u1)
    #u_l_5.append(dice_u5)
    #u_l_10.append(dice_u10)



all_data["Gaussian"] = ga_l
all_data["Hessian"] = h_l
all_data["Laplacian"] = la_l
all_data["Ilastik"] = il_l

all_data["MitoNet"] = u_l_1
#all_data["unet5"] = u_l_5
#all_data["unet10"] = u_l_10

#all_data.to_csv(path + "/dice_coefficient_table.csv")


x = [0,1]
y = [1,1]


# pos_y and pos_x determine position of bar, p sets the number of asterisks, y_dist sets y distance of the asterisk to
# bar, and distance sets the distance between two or more asterisks
#significance_bar(pos_y=1, pos_x=[0, 4], bar_y=0.03, p=1, y_dist=0.02, distance=0.1)

#sb.boxplot(data=all_data).set(ylabel="Dice coefficient")
#plt.show()

sb.distplot(ga_l, color="blue", label="Gaussian", hist=False)
sb.distplot(h_l, color="orange", label="Hessian", hist=False)
sb.distplot(la_l, color="green", label="Laplacian", hist=False)
sb.distplot(il_l, color="red", label="Ilastik", hist=False)
sb.distplot(u_l_1, color="purple", label="MitoNet", hist=False).set(ylabel="Dice coefficient distribution",
                                                                  xlabel="Dice coefficient")
#sb.distplot(u_l_5, color="black", label="unet5", hist=False)
#sb.distplot(u_l_10, color="red", label="unet10", hist=False)


plt.show()


unet = u_l_1

"""
print(normaltest(h_l)[1])
print(normaltest(la_l)[1])
print(normaltest(ga_l)[1])
print(normaltest(il_l)[1])
print(normaltest(u_l_1)[1])
print(normaltest(u_l_5)[1])
print(normaltest(u_l_10)[1])
"""

print(ttest_ind(unet, il_l)[1])
print(mannwhitneyu(unet, il_l)[1])









