"""

15/11/18

Morphological value distribution comparison

Measuring the distribution of area, eccentricity, aspect ratio, perimeter and solidity (of segmented objects) and
checking if distributions in segmentation predictions are significantly different when compared to the distribution in
the ground truth images


"""



import cv2
from skimage.measure import regionprops, label
import os
from scipy.stats.mstats import mannwhitneyu, normaltest, ttest_ind
from scipy.stats import energy_distance
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter("ignore", UserWarning)


# Section 1
###################################################################################################
###################################################################################################

def cohens_d(data1, data2):

    p_std = np.sqrt(((len(data1)-1)*np.var(data1)+(len(data2)-1)*np.var(data2))/(len(data1)+len(data2)-2))

    cohens_d = np.abs(np.average(data1) - np.average(data2)) / p_std

    return cohens_d

path = "C:/Users/Christian/Desktop/Fourth_CV/Complete_images/"
#path = "C:/Users/Christian/Desktop/Third_CV/Image_sections/sections/"

# Get image data
#################################

def morph_distribution(path, seg_name):


    gt_folder_path = path + "Ground_Truth"
    seg_folder_path = path + seg_name

    gt_image_list = os.listdir(gt_folder_path)
    seg_image_list = os.listdir(seg_folder_path)

    columns = ["Image", "Area", "Eccentricity", "Aspect Ratio", "Perimeter", "Solidity"]

    df = pd.DataFrame(columns=columns)
    df_2 = pd.DataFrame(columns=columns)


    list_area = []
    list_ecc = []
    list_ar = []
    list_per = []
    list_sol = []

    list_area2 = []
    list_ecc2 = []
    list_ar2 = []
    list_per2 = []
    list_sol2 = []

    def significance(pvalue):

        #"""
        if pvalue > 0.05:
            return 0

        if 0.01 < pvalue < 0.05:
            return 1

        if 0.001 < pvalue < 0.01:
            return 2

        if pvalue < 0.001:
            return 3
        #"""

        return pvalue

    ###################################################

    ###################################################

    for gt_image, seg_image in zip(gt_image_list, seg_image_list):


        print(gt_image)

        gt = cv2.imread(gt_folder_path + "/" + gt_image, cv2.IMREAD_GRAYSCALE)
        seg = cv2.imread(seg_folder_path + "/" + seg_image, cv2.IMREAD_GRAYSCALE)

        # label image mask
        gt_labelled = label(gt)
        seg_labelled = label(seg)


        # Get region props of labelled images
        gt_reg_props = regionprops(gt_labelled)
        seg_reg_props = regionprops(seg_labelled)


        # compare shape descriptor distributions
        #################################

        # Area
        gt_area = [i.area for i in gt_reg_props]
        seg_area = [i.area for i in seg_reg_props]

        pvalue_area = mannwhitneyu(gt_area, seg_area)[1]

        list_area.append(significance(pvalue_area))
        #list_area.append(pvalue_area)

        list_area2.append(cohens_d(gt_area, seg_area))


        # Eccentricity
        gt_ecc = [i.eccentricity for i in gt_reg_props]
        seg_ecc = [i.eccentricity for i in seg_reg_props]

        pvalue_ecc = mannwhitneyu(gt_ecc, seg_ecc)[1]

        list_ecc.append(significance(pvalue_ecc))
        #list_ecc.append(pvalue_ecc)

        list_ecc2.append(cohens_d(gt_ecc, seg_ecc))

        # Aspect ratio


        gt_ar = [i.major_axis_length/i.minor_axis_length for i in gt_reg_props if i.minor_axis_length != 0]
        seg_ar = [i.major_axis_length/i.minor_axis_length for i in seg_reg_props if i.minor_axis_length != 0]

        pvalue_ar = mannwhitneyu(gt_ar, seg_ar)[1]

        list_ar.append(significance(pvalue_ar))
        #list_ar.append(pvalue_ar)

        list_ar2.append(cohens_d(gt_ar, seg_ar))

        # Perimeter
        gt_per = [i.perimeter for i in gt_reg_props]
        seg_per = [i.perimeter for i in seg_reg_props]

        pvalue_per = mannwhitneyu(gt_per, seg_per)[1]

        list_per.append(significance(pvalue_per))
        #list_per.append(pvalue_per)

        list_per2.append(cohens_d(gt_per, seg_per))

        # Solidity
        gt_sol = [i.solidity for i in gt_reg_props]
        seg_sol = [i.solidity for i in seg_reg_props]

        #print(len(gt_sol))

        pvalue_sol = mannwhitneyu(gt_sol, seg_sol)[1]

        list_sol.append(significance(pvalue_sol))
        #list_sol.append(pvalue_sol)

        list_sol2.append(cohens_d(gt_sol, seg_sol))

        #################################

        def show(gt, seg):

            sb.kdeplot(gt, color="green", shade=True)
            sb.kdeplot(seg, color="red", shade=True)
            plt.show()

        def norm(gt, seg):

            print(normaltest(gt)[1])
            print(normaltest(seg)[1])


        #norm(gt_area, seg_area)
        #norm(gt_ecc, seg_ecc)
        #norm(gt_ar, seg_ar)
        #norm(gt_per, seg_per)
        #norm(gt_sol, seg_sol)

        """
        #show(gt_area, seg_area)
        #print(pvalue_area)
        print(energy_distance(gt_area, seg_area))
        #show(gt_ecc, seg_ecc)
        #print(pvalue_ecc)
        print(energy_distance(gt_ecc, seg_ecc))
        #show(gt_ar, seg_ar)
        #print(pvalue_ar)
        print(energy_distance(gt_ar, seg_ar))
        #show(gt_per, seg_per)
        #print(pvalue_per)
        print(energy_distance(gt_per, seg_per))
        #show(gt_sol, seg_sol)
        #print(pvalue_sol)
        print(energy_distance(gt_sol, seg_sol))
        """



    df["Image"] = gt_image_list
    df["Area"] = list_area
    df["Eccentricity"] = list_ecc
    df["Aspect Ratio"] = list_ar
    df["Perimeter"] = list_per
    df["Solidity"] = list_sol

    df_2["Image"] = gt_image_list
    df_2["Area"] = list_area2
    df_2["Eccentricity"] = list_ecc2
    df_2["Aspect Ratio"] = list_ar2
    df_2["Perimeter"] = list_per2
    df_2["Solidity"] = list_sol2

    total_values = len(gt_image_list)*5

    zero_p = 0
    one_p = 0
    two_p = 0
    three_p = 0

    for index, row in df.iterrows():

        for column in row:

            if column == 0:
                zero_p+=1

            elif column == 1:
                one_p+=1

            elif column == 2:
                two_p+=1

            elif column == 3:
                three_p+=1


    #print(zero_p/total_values)
    #print(one_p/total_values)
    #print(two_p/total_values)
    #print(three_p/total_values)



    # raw data
    df.to_csv(path + seg_name + "_Morph_Dist_comparison.csv")
    #df_2.to_csv(path + seg_name + "_effect_size_Morph_Dist_comparison.csv")


###################################################################################################
###################################################################################################

# morph distribution comparison

seg_name = "MitoSegNet"
morph_distribution(path, seg_name)


seg_name = "Fiji_U-Net"
morph_distribution(path, seg_name)
seg_name = "Gaussian"
morph_distribution(path, seg_name)
seg_name = "Laplacian"
morph_distribution(path, seg_name)
seg_name = "Hessian"
morph_distribution(path, seg_name)
seg_name = "Ilastik"
morph_distribution(path, seg_name)
