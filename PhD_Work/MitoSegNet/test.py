
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from scipy import ndimage as ndi
import cv2
import os
import copy

from skimage.morphology import watershed, remove_small_objects
from skimage.measure import label, regionprops
from skimage.feature import peak_local_max
from skimage.segmentation import mark_boundaries


import warnings
warnings.simplefilter("ignore", UserWarning)


"""
path = "C:/Users/Christian/Desktop/Fourth_CV/Complete_images/MitoSegNet"
path_org = "C:/Users/Christian/Desktop/Fourth_CV/Complete_images/Original"

img_list = os.listdir(path)

thresh = cv2.imread(path + os.sep + img_list[0], cv2.IMREAD_GRAYSCALE)
img = cv2.imread(path_org + os.sep + img_list[0], cv2.IMREAD_GRAYSCALE)

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

cv2.imshow("img", dist_transform)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv2.watershed(img, markers)
"""


path = "C:/Users/Christian/Desktop/Fourth_CV/Complete_images/MitoSegNet"
path_gt = "C:/Users/Christian/Desktop/Fourth_CV/Complete_images/Ground_Truth"


for img in os.listdir(path):

    print(img)

    image = cv2.imread(path + os.sep + img, cv2.IMREAD_GRAYSCALE)

    image_copy = copy.copy(image)
    org_img = copy.copy(image)

    gt = cv2.imread(path_gt + os.sep + img, cv2.IMREAD_GRAYSCALE)

    # label image mask
    labelled = label(image)
    # Get region props of labelled images
    reg_props = regionprops(labelled)

    labelled_gt = label(gt)
    # Get region props of labelled images
    reg_props_gt = regionprops(labelled_gt)

    area_gt = [i.area for i in reg_props_gt]
    #y, x

    area = [i.area for i in reg_props]
    per = [i.perimeter for i in reg_props]
    solidity = [i.solidity for i in reg_props]
    ecc = [i.eccentricity for i in reg_props]


    #plt.hist(area)
    #plt.show()

    #print(np.quantile(area, 0.5))
    #print(np.quantile(solidity, 0.8))
    #print(np.median(area))

    #print(4*np.pi*(np.median(area)/np.power(np.median(per),2)))

    #print(np.median(ecc))
    #if np.median(ecc) < 0.9:
    #    print("Fragmented")

    # removing objects which should not undergo watershed segmentation

    if np.median(ecc) < 0.9:

        for n in reg_props:

            #print(n.coords)

            aspect_ratio = n.major_axis_length/n.minor_axis_length

            #if n.solidity < sol and n.area > ar:

            circ = 4*np.pi*(n.area/np.power(n.perimeter,2))


            # remove all objects that should not undergo watershed
            if n.area > 400 or aspect_ratio > 4 or n.solidity > 0.9:

            #if n.area > np.quantile(area, 0.5) or n.solidity > np.quantile(solidity, 0.8):
            #if n.area > 400 and n.solidity > 0.3:

                for yx in n.coords:
                    image[yx[0], yx[1]] = 0


            #in second image, all those objects that do undergo watershed  are removed
            else:

                for yx in n.coords:
                    image_copy[yx[0], yx[1]] = 0


        #cv2.imshow(img, image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()


        # show objects that will undergo watershed



        # watershed segmentation

        # Generate the markers as local maxima of the distance to the background
        distance = ndi.distance_transform_edt(image)


        local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((8, 8)), labels=image)


        markers = ndi.label(local_maxi)[0]

        labels = watershed(-distance, markers, mask=image)

        labels = labels.astype("uint16")


        reg_props_ws = regionprops(labels)
        area_ws = [i.area for i in reg_props_ws]


        image_copy_label = label(image_copy)

        image_copy_label = image_copy_label.astype("uint16")

        image_copy_label[image_copy_label != 0] += len(area_ws)



        final_img = cv2.add(labels, image_copy_label.astype("uint16"))



        final_img_temp = mark_boundaries(org_img, final_img)

        # selecting image that contains separated objects
        final_img_boundaries = final_img_temp[:, :, 2]

        label_final = label(final_img_boundaries)

        # allow user to specify what minimum object size should be (originally set to 10)
        new_image = remove_small_objects(label_final, 10)


        new_image[new_image > 0] = 255

        cv2.imwrite(img, new_image)


        new_image_label = label(new_image)
        rp = regionprops(new_image_label)


        len_final = [i.label for i in rp]

        #final_img_labels = label(final_img)
        # Get region props of final images
        #reg_props_img = regionprops(image_copy_label)


        #area_img = [i.area for i in reg_props_img]


        print(len(area), len(len_final), len(area_gt))


        dev_old =  np.abs(len(area_gt)-len(area))/len(area_gt)
        dev_new = np.abs(len(area_gt)-len(len_final))/len(area_gt)

        dev_change = dev_new - dev_old


        print(dev_change)

        #print(dev_old, dev_new)

        #cv2.imwrite(img + "_watershed.tif", labels.astype("uint16"))

















"""
import cv2
import numpy as np
import os
from screeninfo import get_monitors
from tkinter import *
from tkinter import messagebox



y = 1030
x = 1300


path = "C:/Users/Christian/Desktop/Third_CV/Complete_images"

arr = cv2.imread(path + os.sep + "Original" + os.sep + "160819 MD3011 spg7+II5D14 RNAi w5.tif")
arr2 = cv2.imread(path + os.sep + "Ground_Truth" + os.sep + "160819 MD3011 spg7+II5D14 RNAi w5.tif")


##########
screen_res = str(get_monitors()[0])
screen_res = (screen_res.split("(")[1])

x_res = int(screen_res.split("x")[0])

y_res = screen_res.split("x")[1]
y_res = int(y_res.split("+")[0])

print(x_res, y_res)
##########

cv2.namedWindow('arr', cv2.WINDOW_NORMAL)
cv2.namedWindow('arr2', cv2.WINDOW_NORMAL)

f = 2.5*(x/x_res)

# x , y
cv2.resizeWindow('arr', int(x/f), int(y/f))
cv2.moveWindow("arr", int(0.1*x_res), int(0.1*y_res))

cv2.resizeWindow('arr2', int(x/f), int(y/f))
cv2.moveWindow("arr2", int(0.5*x_res), int(0.1*y_res))

cv2.imshow("arr", arr)
cv2.imshow("arr2", arr2)



Tk().withdraw()
answer = messagebox.askokcancel("Image selection", "Save predicted segmentation?")

print(answer)

cv2.waitKey(0)
cv2.destroyAllWindows()
"""

