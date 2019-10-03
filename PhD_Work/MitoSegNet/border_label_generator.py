import cv2
import os
import numpy as np

path = "C:/Users/Christian/Desktop/Fourth_CV/Complete_images"


gt_imgs = os.listdir(path + os.sep + "Ground_Truth")
wm_imgs = os.listdir(path + os.sep + "Weight_maps")

for gt, wm in zip(gt_imgs, wm_imgs):

    read_gt = cv2.imread(path + os.sep + "Ground_Truth" + os.sep + gt, cv2.IMREAD_GRAYSCALE)
    read_wm = cv2.imread(path + os.sep + "Weight_maps" + os.sep + wm, cv2.IMREAD_GRAYSCALE)


    read_gt = np.invert(read_gt)
    read_gt = np.divide(read_gt, 255)


    read_wm[read_wm <= 1] = 0
    read_wm[read_wm > 1] = 255
    #read_wm = np.divide(read_wm, 255)

    final_mask = np.multiply(read_wm, read_gt)

    #cv2.imshow(gt, final_mask)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    cv2.imwrite(path + os.sep + "Border_labels" + os.sep + gt, final_mask)