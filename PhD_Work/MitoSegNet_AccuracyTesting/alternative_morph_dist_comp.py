"""

22/07/19

Including intensity and branch descriptor values in morphological comparison

Allows to generate morphological comparison p-value tables and energy distance tables for subsequent analysis

"""

import itertools
import cv2
import os
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import collections

from Plot_Significance import significance_bar
from scipy.stats.mstats import mannwhitneyu, normaltest, ttest_ind
from scipy.stats import energy_distance
from skimage.measure import regionprops, label
from skimage import img_as_bool, io, color
from skimage.morphology import skeletonize
from skan import summarise

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

path = "C:/Users/Christian/Desktop/Fourth_CV/Complete_images"


# Section 1
###################################################################################################
###################################################################################################

def cohens_d(data1, data2):

    p_std = np.sqrt(((len(data1)-1)*np.var(data1)+(len(data2)-1)*np.var(data2))/(len(data1)+len(data2)-2))

    cohens_d = np.abs(np.average(data1) - np.average(data2)) / p_std

    return cohens_d

# Get image data
#################################

def morph_distribution(path, seg_name):


    gt_folder_path = path + os.sep + "Ground_Truth"
    seg_folder_path = path + os.sep + seg_name
    org_folder_path = path + os.sep + "Original"

    #image_list = os.listdir(gt_folder_path)
    image_list = ["170412 MD4049 w10 FL200.tif"]

    columns = ["Image", "Area", "Eccentricity", "Aspect Ratio", "Perimeter", "Solidity", "Number of branches",
               "Branch length", "Total branch length", "Curvature index", "Mean intensity"]

    df = pd.DataFrame(columns=columns)
    df_2 = pd.DataFrame(columns=columns)
    df_c = pd.DataFrame(columns=columns)


    list_area = []
    list_ecc = []
    list_ar = []
    list_per = []
    list_sol = []

    list_nb = []
    list_tbl = []
    list_bl = []
    list_ci = []

    list_int = []


    clist_area = []
    clist_ecc = []
    clist_ar = []
    clist_per = []
    clist_sol = []

    clist_nb = []
    clist_tbl = []
    clist_bl = []
    clist_ci = []

    clist_int = []


    elist_area = []
    elist_ecc = []
    elist_ar = []
    elist_per = []
    elist_sol = []

    elist_nb = []
    elist_tbl = []
    elist_bl = []
    elist_ci = []

    elist_int = []


    ###################################################

    ###################################################

    for image in image_list:


        print(image)

        gt = cv2.imread(gt_folder_path + os.sep + image, cv2.IMREAD_GRAYSCALE)
        seg = cv2.imread(seg_folder_path + os.sep + image , cv2.IMREAD_GRAYSCALE)
        org = cv2.imread(org_folder_path + os.sep + image, cv2.IMREAD_GRAYSCALE)

        # skeletonize
        ##################################################
        ##################################################

        def get_branch_meas(path):

            """
            05-07-19

            for some reason the sumarise function of skan prints out a different number of objects, which is why
            i currently cannot include the branch data in the same table as the morph parameters
            """

            read_lab_skel = img_as_bool(color.rgb2gray(io.imread(path)))
            lab_skel = skeletonize(read_lab_skel).astype("uint8")

            branch_data = summarise(lab_skel)

            # curv_ind = (branch_data["branch-distance"] - branch_data["euclidean-distance"]) / branch_data["euclidean-distance"]

            curve_ind = []
            for bd, ed in zip(branch_data["branch-distance"], branch_data["euclidean-distance"]):

                if ed != 0.0:
                    curve_ind.append((bd - ed) / ed)
                else:
                    curve_ind.append(bd - ed)

            branch_data["curvature-index"] = curve_ind

            grouped_branch_data_mean = branch_data.groupby(["skeleton-id"], as_index=False).mean()

            grouped_branch_data_sum = branch_data.groupby(["skeleton-id"], as_index=False).sum()

            counter = collections.Counter(branch_data["skeleton-id"])

            n_branches = []
            for i in grouped_branch_data_mean["skeleton-id"]:
                n_branches.append(counter[i])

            branch_len = grouped_branch_data_mean["branch-distance"].tolist()
            tot_branch_len = grouped_branch_data_sum["branch-distance"].tolist()
            curv_ind = grouped_branch_data_mean["curvature-index"].tolist()

            return n_branches, branch_len, tot_branch_len, curv_ind

        def significance(pvalue):

            if pvalue > 0.05:
                return 0

            if 0.01 < pvalue < 0.05:
                return 1

            if 0.001 < pvalue < 0.01:
                return 2

            if pvalue < 0.001:
                return 3

            return pvalue

        # pooled standard deviation for calculation of effect size (cohen's d)
        def cohens_d(data1, data2):

            p_std = np.sqrt(
                ((len(data1) - 1) * np.var(data1) + (len(data2) - 1) * np.var(data2)) / (len(data1) + len(data2) - 2))

            cohens_d = np.abs(np.median(data1) - np.median(data2)) / p_std

            return cohens_d


        ##################################################
        ##################################################

        gt_nb, gt_bl, gt_tbl, gt_ci = get_branch_meas(gt_folder_path + os.sep + image)
        seg_nb, seg_bl, seg_tbl, seg_ci = get_branch_meas(seg_folder_path + os.sep + image)


        #######################

        pvalue_nb = mannwhitneyu(gt_nb, seg_nb)[1]
        list_nb.append(significance(pvalue_nb))

        elist_nb.append(energy_distance(gt_nb, seg_nb))
        clist_nb.append(cohens_d(gt_nb, seg_nb))

        pvalue_bl = mannwhitneyu(gt_bl, seg_bl)[1]
        list_bl.append(significance(pvalue_bl))

        elist_bl.append(energy_distance(gt_bl, seg_bl))
        clist_bl.append(cohens_d(gt_bl, seg_bl))

        pvalue_tbl = mannwhitneyu(gt_tbl, seg_tbl)[1]
        list_tbl.append(significance(pvalue_tbl))

        elist_tbl.append(energy_distance(gt_tbl, seg_tbl))
        clist_tbl.append(cohens_d(gt_tbl, seg_tbl))

        pvalue_ci = mannwhitneyu(gt_ci, seg_ci)[1]
        list_ci.append(significance(pvalue_ci))

        elist_ci.append(energy_distance(gt_ci, seg_ci))
        clist_ci.append(cohens_d(gt_ci, seg_ci))

        #######################

        # label image mask
        gt_labelled = label(gt)
        seg_labelled = label(seg)


        # Get region props of labelled images
        gt_reg_props = regionprops(label_image=gt_labelled, intensity_image=org, coordinates='xy')
        seg_reg_props = regionprops(label_image=seg_labelled, intensity_image=org, coordinates='xy')


        # compare shape descriptor distributions
        #################################

        # Intensity

        gt_int = [i.mean_intensity for i in gt_reg_props]
        seg_int = [i.mean_intensity for i in seg_reg_props]

        pvalue_int = mannwhitneyu(gt_int, seg_int)[1]
        list_int.append(significance(pvalue_int))

        elist_int.append(energy_distance(gt_int, seg_int))
        clist_int.append(cohens_d(gt_int, seg_int))

        # Area
        gt_area = [i.area for i in gt_reg_props]
        seg_area = [i.area for i in seg_reg_props]

        pvalue_area = mannwhitneyu(gt_area, seg_area)[1]

        list_area.append(significance(pvalue_area))
        #list_area.append(pvalue_area)
        #list_area2.append(cohens_d(gt_area, seg_area))

        elist_area.append(energy_distance(gt_area, seg_area))
        clist_area.append(cohens_d(gt_area, seg_area))

        # Eccentricity
        gt_ecc = [i.eccentricity for i in gt_reg_props]
        seg_ecc = [i.eccentricity for i in seg_reg_props]

        pvalue_ecc = mannwhitneyu(gt_ecc, seg_ecc)[1]

        list_ecc.append(significance(pvalue_ecc))
        #list_ecc.append(pvalue_ecc)

        #list_ecc2.append(cohens_d(gt_ecc, seg_ecc))

        elist_ecc.append(energy_distance(gt_ecc, seg_ecc))
        clist_ecc.append(cohens_d(gt_ecc, seg_ecc))


        # Aspect ratio

        gt_ar = [i.major_axis_length/i.minor_axis_length for i in gt_reg_props if i.minor_axis_length != 0]
        seg_ar = [i.major_axis_length/i.minor_axis_length for i in seg_reg_props if i.minor_axis_length != 0]

        pvalue_ar = mannwhitneyu(gt_ar, seg_ar)[1]

        list_ar.append(significance(pvalue_ar))
        #list_ar.append(pvalue_ar)

        #list_ar2.append(cohens_d(gt_ar, seg_ar))

        elist_ar.append(energy_distance(gt_ar, seg_ar))
        clist_ar.append(cohens_d(gt_ar, seg_ar))


        # Perimeter
        gt_per = [i.perimeter for i in gt_reg_props]
        seg_per = [i.perimeter for i in seg_reg_props]

        pvalue_per = mannwhitneyu(gt_per, seg_per)[1]

        list_per.append(significance(pvalue_per))
        #list_per.append(pvalue_per)

        #list_per2.append(cohens_d(gt_per, seg_per))

        elist_per.append(energy_distance(gt_per, seg_per))
        clist_per.append(cohens_d(gt_per, seg_per))

        # Solidity
        gt_sol = [i.solidity for i in gt_reg_props]
        seg_sol = [i.solidity for i in seg_reg_props]

        #print(len(gt_sol))

        pvalue_sol = mannwhitneyu(gt_sol, seg_sol)[1]

        list_sol.append(significance(pvalue_sol))
        #list_sol.append(pvalue_sol)

        #list_sol2.append(cohens_d(gt_sol, seg_sol))

        elist_sol.append(energy_distance(gt_sol, seg_sol))
        clist_sol.append(cohens_d(gt_sol, seg_sol))

        #################################


    columns = ["Image", "Area", "Eccentricity", "Aspect Ratio", "Perimeter", "Solidity", "Number of branches",
               "Branch length", "Total branch length", "Curvature index", "Mean Intensity"]


    df["Image"] = image_list
    df["Area"] = list_area
    df["Eccentricity"] = list_ecc
    df["Aspect Ratio"] = list_ar
    df["Perimeter"] = list_per
    df["Solidity"] = list_sol
    df["Number of branches"] = list_nb
    df["Branch length"] = list_bl
    df["Total branch length"] = list_tbl
    df["Curvature index"] = list_ci
    df["Mean intensity"] = list_int

    df_2["Image"] = image_list
    df_2["Area"] = elist_area
    df_2["Eccentricity"] = elist_ecc
    df_2["Aspect Ratio"] = elist_ar
    df_2["Perimeter"] = elist_per
    df_2["Solidity"] = elist_sol
    df_2["Number of branches"] = elist_nb
    df_2["Branch length"] = elist_bl
    df_2["Total branch length"] = elist_tbl
    df_2["Curvature index"] = elist_ci
    df_2["Mean intensity"] = elist_int

    df_2["Image"] = image_list
    df_c["Area"] = clist_area
    df_c["Eccentricity"] = clist_ecc
    df_c["Aspect Ratio"] = clist_ar
    df_c["Perimeter"] = clist_per
    df_c["Solidity"] = clist_sol
    df_c["Number of branches"] = clist_nb
    df_c["Branch length"] = clist_bl
    df_c["Total branch length"] = clist_tbl
    df_c["Curvature index"] = clist_ci
    df_c["Mean intensity"] = clist_int


    #total_values = len(image_list)*5

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

    df.to_csv(path + os.sep + seg_name + "_Morph_Dist_comparison.csv")
    df_2.to_csv(path + os.sep + seg_name + "_EnergyDistance.csv")
    df_c.to_csv(path + os.sep + seg_name + "_EffectSize.csv")

###################################################################################################
###################################################################################################




# morph distribution comparison






#name_l = ["_Morph_Dist_comparison", "_EnergyDistance"]
#name = name_l[1]

seg_name = "MitoSegNet"
morph_distribution(path, seg_name)

exit()

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
















