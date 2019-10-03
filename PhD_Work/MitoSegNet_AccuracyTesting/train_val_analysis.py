"""

15/11/18 - 29/11/18

checking the history csv files for each run during CV3 or MitoNet

"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


filename = "Final_MitoSegNet_656_training_log.csv"
path = "C:/Users/Christian/Desktop/Fourth_CV/Validation_data/" + filename


table = pd.read_csv(path)


#sb.lineplot(x=20*[5], y=l)
#p1 = plt.axvline(x=5, color="black", linewidth=3)
p2 = sb.lineplot(x=list(range(1,21)), y=table["dice_coefficient"], color="red", linewidth=3)
p3 = sb.lineplot(x=list(range(1,21)), y=table["val_dice_coefficient"], color="blue", linewidth=3)


#.set(xlabel="Epoch", ylabel="Dice coefficient")

#plt.legend((p1, p2, p3), ("Minimum validation loss", "Training", "Validation"),
#           prop={"size": 10}, bbox_to_anchor=(1, 0.5))



p3.set_ylabel("Dice coefficient", fontsize=32)
p3.set_xlabel("Epochs", fontsize=32)



p3.tick_params(axis="x", labelsize=26)

#xint = [0, 5, 10, 15, 20]
#plt.xticks(xint)

p3.tick_params(axis="y", labelsize=26)

#plt.legend(('Minimum validation loss', 'Training', 'Validation'), prop={"size": 26}, loc="upper right")
plt.legend(('Training', 'Validation'), prop={"size": 26}, loc="lower right")

#plt.legend(('Minimum validation loss', 'Training', 'Validation'), prop={"size": 20},
#           bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.margins(x=0)

plt.show()

