"""

testing skeletonize and subsequent branch analysis

"""


from skimage import img_as_bool, io, color
from skimage.morphology import skeletonize, medial_axis
from skan import skeleton_to_csgraph
from skan import summarise

import collections
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)

path = "C:/Users/Christian/Desktop/Third_CV/Complete_images/MitoSegNet/170412 MD4046 w1 next to vulva FL200.tif"
#path = "skeletonize_examples/5.tif"

# reads the image, converts from rgb to grayscale, converts to boolean array
image = img_as_bool(color.rgb2gray(io.imread(path)))
out = skeletonize(image).astype("uint8")

#print(type(out))
#io.imsave("test2.tif", out)

pixel_graph, coordinates, degrees = skeleton_to_csgraph(out)

branch_data = summarise(out)

print(branch_data)

grouped_branch_data_mean = branch_data.groupby(["skeleton-id"], as_index=False).mean()
grouped_branch_data_sum = branch_data.groupby(["skeleton-id"], as_index=False).sum()

sum_table = pd.DataFrame(columns=["Number of branches", "Average branch length", "Total object length", "Average curvature index"])

counter = collections.Counter(branch_data["skeleton-id"])

n_branches = []
for i in grouped_branch_data_mean["skeleton-id"]:
    n_branches.append(counter[i])

n_branches_check = []
for i in grouped_branch_data_mean["skeleton-id"]:
    n_branches_check.append(i)

"""
print(grouped_branch_data_sum)
print(grouped_branch_data_mean)
print(n_branches_check)
print(n_branches)
"""

sum_table["Number of branches"] = n_branches
sum_table["Average branch length"] = grouped_branch_data_mean["branch-distance"]
sum_table["Total object length"] = grouped_branch_data_sum["branch-distance"]
sum_table["Average curvature index"] = [((bd-ed)/ed) for bd, ed in zip(grouped_branch_data_mean["branch-distance"],
                                                                       grouped_branch_data_mean["euclidean-distance"])]

print(sum_table)





#print(grouped_branch_data)
#print(n_branches)


