"""

15/11/18 - 29/11/18

checking the history csv files for each run during CV3 or MitoNet

"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


filename = "MitoNet_Training_History.xlsx"
path = "C:/Users/Christian/Documents/Work/Project_Celegans_Mitochondria_Segmentation/Unet_Segmentation/quantiative_" \
       "segmentation_comparison/Cross_Validation/Third_CV/Validation_data/" + filename


table = pd.read_excel(path)



l = np.linspace(0,0.0,20)
print(l)
print(len(l))

#sb.lineplot(x=20*[5], y=l)
p1 = plt.axvline(x=5, color="black", label="Stopping point")
p2 = sb.lineplot(x=table["epoch"], y=table["dice_coefficient"], label="Training")
p3 = sb.lineplot(x=table["epoch"], y=table["val_dice_coefficient"], label="Validation").set(xlabel="Epoch", ylabel="Dice coefficient")

#plt.legend((p1, p2, p3), ("Minimum validation loss", "Training", "Validation"),
#           prop={"size": 10}, bbox_to_anchor=(1, 0.5))

#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.legend(loc="lower right")

plt.show()

