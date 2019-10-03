from skimage.feature import canny
import cv2
import os

path = "C:/Users/Christian/Desktop/Fourth_CV/Complete_images/MitoSegNet"


image = cv2.imread(path + os.sep + os.listdir(path)[0], cv2.IMREAD_GRAYSCALE)


edges = canny(image, sigma=3, low_threshold=10, high_threshold=50)

print(type(edges), edges.dtype)

#edges[edges > 0] = 255

cv2.imwrite("edges.tif", edges.astype("uint8"))