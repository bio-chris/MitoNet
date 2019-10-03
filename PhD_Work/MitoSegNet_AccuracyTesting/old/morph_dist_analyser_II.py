"""

15/01/19

Analysing p-values of morphological distribution comparison together with cohens d value

"""

import os
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from Plot_Significance import significance_bar
from scipy.stats import chisquare, chi2_contingency


path = "C:/Users/Christian/Documents/Work/Project_Celegans_Mitochondria_Segmentation/Unet_Segmentation/quantiative_" \
       "segmentation_comparison/Cross_Validation/Third_CV/Morph_Distribution_Comparison_with_cohensd"


file_list = os.listdir(path)

# switch list element at position 2 with list element at position 3 (ilastik<>laplacian)
#file_list[2], file_list[3] = file_list[3], file_list[2]

s = []

s1 = []
s2 = []
s3 = []

ns = []


observed_values = []

for file in file_list:



    print(file)

    table = pd.read_csv(path + "/" + file)

    # removing first column
    table.drop(table.columns[[0]], axis=1, inplace=True)

    #print(table)

    pval = []
    coh = []

    #for (index, row), phenotype in zip(table.iterrows(), phenotypes):
    for index, row in table.iterrows():

        if row["p-value"] < 0.05:

            pval.append(row["p-value"])
            coh.append(row["cohens d"])


    print(len(pval))
    print(np.average(pval))
    print(np.average(coh))




exit()




#print(observed_values)

# 0: gaussian, 1: hessian, 2: laplacian, 3: ilastik, 4: mitonet

f_obs_gauss = [observed_values[0], observed_values[-1]]
f_obs_hess = [observed_values[1], observed_values[-1]]
f_obs_lapl = [observed_values[2], observed_values[-1]]
f_obs_ilast = [observed_values[3], observed_values[-1]]

p_ga = chi2_contingency(f_obs_gauss, correction=False)[1]
p_he = chi2_contingency(f_obs_hess, correction=False)[1]
p_la = chi2_contingency(f_obs_lapl, correction=False)[1]
p_il = chi2_contingency(f_obs_ilast, correction=False)[1]

"""
print("gaussian:", p_ga)
print("hessian:", p_he)
print("laplacian:", p_la)
print("ilastik:", p_il)
"""

def star_counter(pvalue):

    if pvalue > 0.05:
        return 0

    if 0.01 < pvalue < 0.05:
        return 1

    if 0.001 < pvalue < 0.01:
        return 2

    if pvalue < 0.001:
        return 3


print(ns)
print(s)

ind = ["Gaussian", "Hessian", "Laplacian", "Ilastik", "MitoSegNet"]

# ilastik vs mitonet
significance_bar(pos_y=1.1, pos_x=[3, 4], bar_y=0.03, p=star_counter(p_il), y_dist=0.03, distance=0.11)
# laplacian vs mitonet
significance_bar(pos_y=1.2, pos_x=[2, 4], bar_y=0.03, p=star_counter(p_la), y_dist=0.03, distance=0.11)
# hessian vs mitonet
significance_bar(pos_y=1.3, pos_x=[1, 4], bar_y=0.03, p=star_counter(p_he), y_dist=0.03, distance=0.11)
# gaussian vs mitonet
significance_bar(pos_y=1.4, pos_x=[0, 4], bar_y=0.03, p=star_counter(p_ga), y_dist=0.03, distance=0.11)


# creating stacked bar graph
p1 = plt.bar(ind, ns)

s = [1-i for i in ns]

p2 = plt.bar(ind, s, bottom=ns)
#p2 = plt.bar(ind, s1, bottom=ns, color="rosybrown")
#p3 = plt.bar(ind, s2, bottom=[i+j for i,j in zip(ns, s1)], color="firebrick")
#p4 = plt.bar(ind, s3, bottom=[i+j+n for i,j,n in zip(ns, s1, s2)], color="red")



plt.ylabel("Percent of morphological\ndistribution comparisons", size=18)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)

#plt.legend((p1[0], p2[0], p3[0], p4[0]), ('p>0.05', 'p<0.05', 'p<0'), loc="upper left")

plt.legend((p1[0], p2[0]), ('p>0.05', 'p<0.05', '0.001<p<0.01', 'p<0.001'),
           prop={"size": 14}, bbox_to_anchor=(1, 0.5))

#plt.legend((p1[0], p2[0], p3[0], p4[0]), ('p>0.05', '0.01<p<0.05', '0.001<p<0.01', 'p<0.001'),
#           prop={"size": 14}, bbox_to_anchor=(1, 0.5))


plt.show()






