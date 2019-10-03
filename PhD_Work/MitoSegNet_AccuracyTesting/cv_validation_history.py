"""

21/01/19

script to collect all cross validation validation history data into one file

"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

path = "C:/Users/Christian/Desktop/Fourth_CV/Validation_data/Cross_Val"


csv_list = os.listdir(path)

new_table = pd.DataFrame(columns=csv_list)
new_table2 = pd.DataFrame(columns=csv_list)

l = []
l2 = []

for table in csv_list:

    data = pd.read_csv(path + os.sep + table)

    print(table)

    # which parameter to check
    #l.append(data["val_dice_coefficient"].tolist())
    #l2.append(data["dice_coefficient"].tolist())

    #print(data["val_dice_coefficient"].tolist())

    #flat_l = [np.abs(val) for sublist in l for val in sublist]
    #flat_l2 = [np.abs(val) for sublist in l2 for val in sublist]

    #plt.plot(flat_l)

    new_table[table] = data["val_loss"].tolist()
    new_table2[table] = data["loss"].tolist()

average = new_table.mean(axis=1)
max = new_table.max(axis=1)
min = new_table.min(axis=1)
std = new_table.std(axis=1)

max = average + std/2
min = average - std/2

#####
average2 = new_table2.mean(axis=1)
max2 = new_table2.max(axis=1)
min2 = new_table2.min(axis=1)
std2 = new_table2.std(axis=1)

max2 = average2 + std2/2
min2 = average2 - std2/2
#####

print(len(average.values))
print(len(average2.values))


#plt.plot(max, color="blue", linestyle="dashed")
#plt.plot(min, color="blue", linestyle="dashed")
#n = sb.lineplot(average, color="black")

#plt.plot(x=list(range(1,11)), y=average.values, color="blue", linewidth=3)


sb.lineplot(x=list(range(1,16)), y=average.values, color="blue", linewidth=3)
sb.lineplot(x=list(range(1,16)), y=average2.values, color="red", linewidth=3)


plt.fill_between(list(range(1,16)),average.values-std.values,average.values+std.values,alpha=.2, color="blue")
plt.fill_between(list(range(1,16)),average2.values-std2.values,average2.values+std2.values,alpha=.2, color="red")


plt.ylabel("Average loss", fontsize=32)
plt.xlabel("Epochs", fontsize=32)
plt.tick_params(axis="x", labelsize=26)
plt.tick_params(axis="y", labelsize=26)

plt.margins(x=0)



#sb.lineplot(data=new_table, style=None)
plt.legend(["Validation", "Training"], loc="upper right", fontsize=26)
#plt.legend(sub_dirs, bbox_to_anchor=(1, 1),
#           bbox_transform=plt.gcf().transFigure)
plt.show()