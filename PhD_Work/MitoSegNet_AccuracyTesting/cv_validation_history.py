"""

21/01/19

script to collect all cross validation validation history data into one file

"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sb

path = "C:/Users/Christian/Documents/Work/Project_Celegans_Mitochondria_Segmentation/Unet_Segmentation/quantiative_" \
       "segmentation_comparison/Cross_Validation/Third_CV/Validation_data/Cross_Val"


sub_dirs = os.listdir(path)

new_table = pd.DataFrame(columns=sub_dirs)

for folder in sub_dirs:

    table_list = os.listdir(path + "/" + folder)

    l = []

    for table in table_list:

        data = pd.read_csv(path + "/" + folder + "/" + table)

        # which parameter to check
        l.append(data["val_dice_coefficient"].tolist())

    flat_l = [-1*val for sublist in l for val in sublist]

    plt.plot(flat_l)

    new_table[folder] = flat_l


print(new_table)

#sb.lineplot(x=new_table)
#plt.legend(sub_dirs)
plt.legend(sub_dirs, bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)
plt.show()