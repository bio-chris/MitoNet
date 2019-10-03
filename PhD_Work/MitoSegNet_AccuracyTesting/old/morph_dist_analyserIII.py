"""

06/06/19

analysing raw p value data in addition to effect size data to get average effect size per p-value category

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from Plot_Significance import significance_bar
from scipy.stats import chisquare, chi2_contingency

path = "C:/Users/Christian/Desktop/Fourth_CV"

raw_path = path + os.sep + "Morph_Distribution_Comparison_raw"
es_path = path + os.sep + "Morph_Distribution_Comparison_with_cohensd"

raw_file_list = os.listdir(raw_path)
es_file_list = os.listdir(es_path)


name_file_list_raw = ['MitoSegNet_raw_Morph_Dist_comparison.csv', 'Fiji_U-Net_pretrained_raw_Morph_Dist_comparison.csv',
                      'Ilastik_raw_Morph_Dist_comparison.csv', 'Gaussian_raw_Morph_Dist_comparison.csv',
                      'Hessian_raw_Morph_Dist_comparison.csv', 'Laplacian_raw_Morph_Dist_comparison.csv']

name_file_list_es = ['MitoSegNet_effect_size_Morph_Dist_comparison.csv',
                     'Fiji_U-Net_pretrained_effect_size_Morph_Dist_comparison.csv',
                     'Ilastik_effect_size_Morph_Dist_comparison.csv', 'Gaussian_effect_size_Morph_Dist_comparison.csv',
                     'Hessian_effect_size_Morph_Dist_comparison.csv', 'Laplacian_effect_size_Morph_Dist_comparison.csv']

#print(file_list)

ms = []
pf = []
i = []
g = []
h = []
la = []


data_d = {"MitoSegNet": [], "Pretrained\nFiji U-Net": [], "Ilastik": [], "Gaussian": [], "Hessian": [], "Laplacian": []}


for raw_file, es_file, seg in zip(name_file_list_raw, name_file_list_es, data_d):

    #print(raw_file)

    raw_table = pd.read_csv(raw_path + os.sep + raw_file)
    es_table = pd.read_csv(es_path + os.sep + es_file)

    # removing first column
    raw_table.drop(raw_table.columns[[0,1]], axis=1, inplace=True)
    es_table.drop(es_table.columns[[0,1]], axis=1, inplace=True)

    total_values = 60

    zero_p = 0
    one_p = 0
    two_p = 0
    three_p = 0

    l = []

    for (index, row), (index2, row2) in zip(raw_table.iterrows(), es_table.iterrows()):

        for column, column2 in zip(row, row2):

            if column < 0.05:

                l.append(column2)

                data_d[seg].append(column2)


    print(seg)




"""
# converting dictionary with different list lengths into a pandas dataframe

solution from https://stackoverflow.com/questions/19736080/creating-dataframe-from-a-dictionary-where-entries-
# have-different-lengths/32383078
"""
data = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data_d.items()]))


print(data.mean())

n = sb.boxplot(data=data, color="white", fliersize=0)
sb.swarmplot(data=data, color="black")

plt.show()








