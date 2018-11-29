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
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter("ignore", UserWarning)


# Section 1
###################################################################################################
###################################################################################################


path = "C:/Users/Christian/Documents/Work/Project_Celegans_Mitochondria_Segmentation/Unet_Segmentation/quantiative_" \
       "segmentation_comparison/Cross_Validation/Third_CV/Complete_images/"


# Get image data
#################################

def morph_distribution(path, seg_name):


    gt_folder_path = path + "Ground_Truth"
    seg_folder_path = path + seg_name

    gt_image_list = os.listdir(gt_folder_path)
    seg_image_list = os.listdir(seg_folder_path)

    columns = ["Image", "Area", "Eccentricity", "Aspect Ratio", "Perimeter", "Solidity"]

    df = pd.DataFrame(columns= columns)


    list_area = []
    list_ecc = []
    list_ar = []
    list_per = []
    list_sol = []


    def significance(pvalue):
        if pvalue > 0.05:
            return 0

        if 0.01 < pvalue < 0.05:
            return 1

        if 0.001 < pvalue < 0.01:
            return 2

        if pvalue < 0.001:
            return 3


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


        # Eccentricity
        gt_ecc = [i.eccentricity for i in gt_reg_props]
        seg_ecc = [i.eccentricity for i in seg_reg_props]

        pvalue_ecc = mannwhitneyu(gt_ecc, seg_ecc)[1]
        list_ecc.append(significance(pvalue_ecc))

        # Aspect ratio
        gt_ar = [i.major_axis_length/i.minor_axis_length for i in gt_reg_props]
        seg_ar = [i.major_axis_length/i.minor_axis_length for i in seg_reg_props]

        pvalue_ar = mannwhitneyu(gt_ar, seg_ar)[1]
        list_ar.append(significance(pvalue_ar))

        # Perimeter
        gt_per = [i.perimeter for i in gt_reg_props]
        seg_per = [i.perimeter for i in seg_reg_props]

        pvalue_per = mannwhitneyu(gt_per, seg_per)[1]
        list_per.append(significance(pvalue_per))

        # Solidity
        gt_sol = [i.solidity for i in gt_reg_props]
        seg_sol = [i.solidity for i in seg_reg_props]

        pvalue_sol = mannwhitneyu(gt_sol, seg_sol)[1]
        list_sol.append(significance(pvalue_sol))

        #################################

        def show(gt, seg):

            sb.distplot(gt, color="green")
            sb.distplot(seg, color="red")
            plt.show()

        def norm(gt, seg):

            print(normaltest(gt)[1])
            print(normaltest(seg)[1])


        norm(gt_area, seg_area)
        norm(gt_ecc, seg_ecc)
        norm(gt_ar, seg_ar)
        norm(gt_per, seg_per)
        norm(gt_sol, seg_sol)

        #show(gt_area, seg_area)
        #show(gt_ecc, seg_ecc)
        #show(gt_ar, seg_ar)
        #show(gt_per, seg_per)
        #show(gt_sol, seg_sol)



    df["Image"] = gt_image_list
    df["Area"] = list_area
    df["Eccentricity"] = list_ecc
    df["Aspect Ratio"] = list_ar
    df["Perimeter"] = list_per
    df["Solidity"] = list_sol

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
    #df.to_csv(path  + "U-Net10_Morph_Dist_comparison.csv")



###################################################################################################
###################################################################################################

# morph distribution comparison

morph_distribution(path, "U-Net/10_epochs")