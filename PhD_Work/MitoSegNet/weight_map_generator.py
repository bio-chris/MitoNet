
import numpy as np
from skimage.measure import label as set_labels, regionprops
from scipy.ndimage.morphology import distance_transform_edt as get_dmap
import copy
import cv2
import os

def create_distance_weight_map(label, w0=10, sigma=5):

    # creating first parameter of weight map formula
    template_weight_map = np.ones_like(label)
    template_weight_map[label > 0] = 2

    # setting all 255 values to 1
    label[label > 1] = 1
    # inverting label for distance_transform
    new_label = 1 - label

    # calculate distance_transform
    dist_map1 = get_dmap(new_label)

    # labels each separable object with one unique pixel value
    labelled = set_labels(label)

    # creates list with label properties (for us important: coordinates)
    regprops = regionprops(labelled)

    stack = []

    # iterate through every object in image
    for i in regprops:

        # create shallow copy of new_label (modifying matrix, without changing original)
        temp = copy.copy(new_label)

        # iterate through coordinates of each object
        for n in i.coords:
            # create one image each, in which one object is removed (background = 1)
            temp[n[0], n[1]] = 1

        stack.append(get_dmap(temp))

    # create empty matrix
    dist_map2 = np.zeros_like(label)

    x = 0
    # iterate through each row of distance map 1
    for row in dist_map1:

        y = 0
        # iterate through each column
        for col in row:
            for img in stack:

                # check if at position x,y the pixel value of img is bigger than dist_map1 >> distance to second nearest border
                if img[x, y] > dist_map1[x, y]:

                    dist_map2[x, y] = img[x, y]
                    break

                else:
                    dist_map2[x, y] = dist_map1[x, y]

            y += 1
        x += 1

    weight_map = template_weight_map + w0 * np.exp(- ((dist_map1 + dist_map2) ** 2 / (2 * sigma ** 2)))

    return weight_map


path = "C:/Users/Christian/Desktop/Fourth_CV/Complete_images/Ground_Truth"
save_path = "C:/Users/Christian/Desktop/Fourth_CV/Complete_images/Weight_maps"

for image in os.listdir(path):

    print(image)

    read_img = cv2.imread(path + os.sep + image, cv2.IMREAD_GRAYSCALE)

    weight_img = create_distance_weight_map(read_img)

    cv2.imwrite(save_path + os.sep + image, weight_img)