"""

27/11/18

Create a python generated stacked bar figure of the morphological accuracy

"""

import os
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from Plot_Significance import significance_bar
from scipy.stats import chisquare, chi2_contingency


#path = "C:/Users/Christian/Desktop/Third_CV/Morph_Distribution_Comparison"
path = "C:/Users/Christian/Desktop/Fourth_CV/Morph_Distribution_Comparison_Extended"


file_list = os.listdir(path)

# switch list element at position 2 with list element at position 3 (ilastik<>laplacian)
#file_list[2], file_list[3] = file_list[3], file_list[2]
#file_list[6], file_list[5] = file_list[5], file_list[6]

file_list = ['MitoSegNet_Morph_Dist_comparison.csv', 'Fiji_U-Net_Morph_Dist_comparison.csv' ,
             'Ilastik_Morph_Dist_comparison.csv', 'Gaussian_Morph_Dist_comparison.csv', 'Hessian_Morph_Dist_comparison.csv',
             'Laplacian_Morph_Dist_comparison.csv']


#print(file_list)

s = []

s1 = []
s2 = []
s3 = []

ns = []

phenotypes = ["mixed", "mixed", "fragmented", "tubular", "tubular", "tubular", \
             "fragmented", "elongated", "elongated", "elongated", "mixed", "fragmented"]

observed_values = []

for file in file_list:

    if ".csv" in file:

        print(file)

        table = pd.read_csv(path + "/" + file)

        # removing first column
        table.drop(table.columns[[0]], axis=1, inplace=True)

        total_values = 120

        zero_p = 0
        one_p = 0
        two_p = 0
        three_p = 0

        #for (index, row), phenotype in zip(table.iterrows(), phenotypes):
        for (index, row) in table.iterrows():

            #if phenotype == "fragmented":

            for column in row:

                if column == 0:
                    zero_p += 1

                elif column == 1:
                    one_p += 1

                elif column == 2:
                    two_p += 1

                elif column == 3:
                    three_p += 1


        #print(zero_p/total_values)
        #print(one_p/total_values)
        #print(two_p/total_values)
        #print(three_p/total_values)

        ns.append(zero_p/total_values)
        s.append((total_values-zero_p)/total_values)

        s1.append(one_p/total_values)
        s2.append(two_p/total_values)
        s3.append(three_p/total_values)


        print(zero_p)
        print(one_p)
        print(two_p)
        print(three_p)

        observed_values.append([zero_p, one_p+two_p+three_p])


print(observed_values)

# 0: mitosegnet, 1: pretrained fiji unet, 2: ilastik, 3: gaussian, 4: hessian, 5: laplacian

f_obs_gauss = [observed_values[0], observed_values[3]]
f_obs_hess = [observed_values[0], observed_values[4]]
f_obs_lapl = [observed_values[0], observed_values[5]]
f_obs_ilast = [observed_values[0], observed_values[2]]
f_obs_pt_fu = [observed_values[0], observed_values[1]]

p_ga = chi2_contingency(f_obs_gauss, correction=False)[1]
p_he = chi2_contingency(f_obs_hess, correction=False)[1]
p_la = chi2_contingency(f_obs_lapl, correction=False)[1]
p_il = chi2_contingency(f_obs_ilast, correction=False)[1]
p_pt_fu = chi2_contingency(f_obs_pt_fu, correction=False)[1]


print(p_he)
print(p_la)
print(p_ga)
print(p_il)
print(p_pt_fu)
#print("fiji u-net:", p_fu)

def star_counter(pvalue):

    if pvalue > 0.05:
        return 0

    if 0.01 < pvalue < 0.05:
        return 1

    if 0.001 < pvalue < 0.01:
        return 2

    if pvalue < 0.001:
        return 3

# "Gaussian", "Hessian", "Laplacian", "Ilastik", "MitoSegNet", "Pretrained\nFiji U-Net", "Fiji U-Net"

ind = ["MitoSegNet", "Pretrained\nFiji U-Net", "Ilastik", "Gaussian", "Hessian", "Laplacian"]


dist = 0.1
significance_bar(pos_y=1.1, pos_x=[0, 1], bar_y=0.03, p=2, y_dist=0.03, distance=dist)
significance_bar(pos_y=1.2, pos_x=[0, 2], bar_y=0.03, p=2, y_dist=0.03, distance=dist)
significance_bar(pos_y=1.3, pos_x=[0, 3], bar_y=0.03, p=3, y_dist=0.03, distance=dist)
significance_bar(pos_y=1.4, pos_x=[0, 4], bar_y=0.03, p=3, y_dist=0.03, distance=dist)
significance_bar(pos_y=1.5, pos_x=[0, 5], bar_y=0.03, p=3, y_dist=0.03, distance=dist)



# creating stacked bar graph

#p1 = plt.bar(ind, ns, color="white", edgecolor="black")
#p2 = plt.bar(ind, s, bottom=ns, color="black", edgecolor="black")


p1 = plt.bar(ind, ns, color="white", edgecolor="black")
p2 = plt.bar(ind, s, bottom=ns, color="black", edgecolor="black")

#p2 = plt.bar(ind, s, bottom=ns, color="silver", edgecolor="black")
#p3 = plt.bar(ind, s2, bottom=[i+j for i,j in zip(ns, s1)], color="gray", edgecolor="black")
#p4 = plt.bar(ind, s3, bottom=[i+j+n for i,j,n in zip(ns, s1, s2)], color="black", edgecolor="black")



plt.ylabel("p-value frequency", size=32)

#plt.legend((p1[0], p2[0], p3[0], p4[0]), ('p>0.05', 'p<0.05', 'p<0'), loc="upper left")

#plt.legend((p1[0], p2[0], p3[0], p4[0]), ('p>0.05', '0.01<p<0.05', '0.001<p<0.01', 'p<0.001'),
#           prop={"size": 20}, bbox_to_anchor=(1, 0.5))

plt.legend((p1[0], p2[0]), ('p>0.05', 'p<0.05'),
           prop={"size": 20}, bbox_to_anchor=(1, 0.5))

#plt.tick_params(axis="x", labelsize=28, rotation=45)
plt.tick_params(axis="x", labelsize=34, rotation=45)
plt.tick_params(axis="y", labelsize=28)

plt.show()






