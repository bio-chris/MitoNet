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


#path = "C:/Users/Christian/Desktop/Third_CV/Complete_images"
path = "C:/Users/Christian/Desktop/Third_CV/Image_sections/sections"


gt_folder = "Ground_Truth"
gauss_folder = "Gaussian"
hess_folder = "Hessian"
laplac_folder = "Laplacian"
il_folder = "Ilastik"

# trained for 17,600 iterations, roughly equal to 5 epochs for MitoSegNet
#unet_folder = "Fiji_U-Net"

# pretrained using 2d cell image dataset, trained for 1000 or 3000 iterations (roughly 0.28 or 0.85 epochs)
unet_folder_pretrained = "Fiji_U-Net_pretrained"

mitosegnet_folder = "MitoSegNet"


gt_directory = path + "/" + gt_folder
gauss_directory = path + "/" + gauss_folder
hess_dir = path + "/" + hess_folder
laplac_dir = path + "/" + laplac_folder
il_dir = path + "/" + il_folder


mitosegnet_dir = path + "/" + mitosegnet_folder
unet_dir_pretrain = path + "/" + unet_folder_pretrained
#unet_dir = path + "/" + unet_folder

ga_l = []
h_l = []
la_l = []
il_l = []

m_l = []
u_l_pt = []
u_l = []

all_data = pd.DataFrame(columns=["MitoSegNet", "Pretrained\nFiji U-Net", "Ilastik", "Gaussian", "Hessian", "Laplacian"])

for gt in os.listdir(gt_directory):

    #print(gt)

    true = cv2.imread(gt_directory + "/" + gt, cv2.IMREAD_GRAYSCALE)

    pred_gauss = cv2.imread(gauss_directory + "/" + gt, cv2.IMREAD_GRAYSCALE)
    pred_hess = cv2.imread(hess_dir + "/" + gt, cv2.IMREAD_GRAYSCALE)
    pred_laplac = cv2.imread(laplac_dir + "/" + gt, cv2.IMREAD_GRAYSCALE)
    pred_il = cv2.imread(il_dir + "/" + gt, cv2.IMREAD_GRAYSCALE)

    pred_msn = cv2.imread(mitosegnet_dir + "/" + gt, cv2.IMREAD_GRAYSCALE)
    pred_unet_pt = cv2.imread(unet_dir_pretrain + "/" + gt, cv2.IMREAD_GRAYSCALE)
    #pred_unet = cv2.imread(unet_dir + "/" + u, cv2.IMREAD_GRAYSCALE)


    dice_ga = (np.sum(pred_gauss[true == 255]) * 2.0) / (np.sum(pred_gauss) + np.sum(true))
    dice_h = (np.sum(pred_hess[true == 255]) * 2.0) / (np.sum(pred_hess) + np.sum(true))
    dice_l = (np.sum(pred_laplac[true == 255]) * 2.0) / (np.sum(pred_laplac) + np.sum(true))
    dice_il = (np.sum(pred_il[true == 255]) * 2.0) / (np.sum(pred_il) + np.sum(true))

    #print(pred_msn, true)

    dice_msn = (np.sum(pred_msn[true == 255]) * 2.0) / (np.sum(pred_msn) + np.sum(true))
    dice_u_pt = (np.sum(pred_unet_pt[true == 255]) * 2.0) / (np.sum(pred_unet_pt) + np.sum(true))
    #dice_u = (np.sum(pred_unet[true == 255]) * 2.0) / (np.sum(pred_unet) + np.sum(true))

    #print(dice_u)

    ga_l.append(dice_ga)
    h_l.append(dice_h)
    la_l.append(dice_l)
    il_l.append(dice_il)

    m_l.append(dice_msn)
    u_l_pt.append(dice_u_pt)
    #u_l.append(dice_u)



all_data["Gaussian"] = ga_l
all_data["Hessian"] = h_l
all_data["Laplacian"] = la_l
all_data["Ilastik"] = il_l

all_data["MitoSegNet"] = m_l
all_data["Pretrained\nFiji U-Net"] = u_l_pt
#all_data["Fiji U-Net"] = u_l

#all_data.to_csv(path + "/dice_coefficient_table.csv")


x = [0,1]
y = [1,1]


# pos_y and pos_x determine position of bar, p sets the number of asterisks, y_dist sets y distance of the asterisk to
# bar, and distance sets the distance between two or more asterisks
significance_bar(pos_y=1, pos_x=[0, 1], bar_y=0.02, p=1, y_dist=0.02, distance=0.1)
#significance_bar(pos_y=1.1, pos_x=[0, 3], bar_y=0.02, p=1, y_dist=0.02, distance=0.1)

print(np.average(h_l))
print(np.average(la_l))
print(np.average(ga_l))
print(np.average(il_l))
print(np.average(m_l))
print(np.average(u_l_pt))


print("\n")

print(normaltest(h_l)[1])
print(normaltest(la_l)[1])
print(normaltest(ga_l)[1])
print(normaltest(il_l)[1])
print(normaltest(m_l)[1])
print(normaltest(u_l_pt)[1])
#print(normaltest(u_l)[1])

print("\n")

print(mannwhitneyu(m_l, h_l)[1])
print(mannwhitneyu(m_l, la_l)[1])
print(mannwhitneyu(m_l, ga_l)[1])
print(mannwhitneyu(m_l, il_l)[1])
print(mannwhitneyu(m_l, u_l_pt)[1])
#print(mannwhitneyu(m_l, u_l)[1])

n = sb.boxplot(data=all_data, color="white", fliersize=0)
sb.swarmplot(data=all_data, color="black")
n.set_ylabel("Dice coefficient", fontsize=32)
n.tick_params(axis="x", labelsize=28, rotation=45)
n.tick_params(axis="y", labelsize=28)

#plt.xticks(labelsize=22, rotation=45)
plt.show()

exit()

"""
sb.distplot(ga_l, color="blue", label="Gaussian", hist=False)
sb.distplot(h_l, color="orange", label="Hessian", hist=False)
sb.distplot(la_l, color="green", label="Laplacian", hist=False)
sb.distplot(il_l, color="red", label="Ilastik", hist=False)
sb.distplot(u_l_1, color="purple", label="MitoNet", hist=False).set(ylabel="Dice coefficient distribution",
                                                                  xlabel="Dice coefficient")
#sb.distplot(u_l_5, color="black", label="unet5", hist=False)
#sb.distplot(u_l_10, color="red", label="unet10", hist=False)
"""

plt.show()













