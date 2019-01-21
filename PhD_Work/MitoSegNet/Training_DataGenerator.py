"""

Partially based on
https://github.com/zhixuhao/unet

>>INSERT DESCRIPTION HERE<<

Written for usage on pcd server

Christian Fischer

---

Notes:

09/11/18

Finalised version of the original data.py script

Two classes:

myAugmentation

Read train and label images separately and merge them
Using Keras preprocessing to augment the merged image
Separate augmented image back into single train and label image

dataProcess

create train and test data
load train and test data


29/03/18

No weight map is used

"""

import numpy as np
import os
import shutil
import glob
import cv2
import re
from keras.preprocessing.image import ImageDataGenerator
from skimage.measure import label as set_labels, regionprops
from scipy.ndimage.morphology import distance_transform_edt as get_dmap


class preprocess(object):

    def __init__(self, train_path="data/train/image", label_path="data/train/label", raw_path="data/train/RawImgs",
                 img_type="tif"):

        """
        Using glob to get all .img_type form path
        """

        self.train_imgs = glob.glob(train_path + "/*." + img_type)
        self.label_imgs = glob.glob(label_path + "/*." + img_type)
        self.train_path = train_path
        self.raw_path = raw_path
        self.label_path = label_path

    def splitImgs(self):

        """
        split original images (1300x1030) into 4 tiles of 656x656
        """

        path_train = self.train_path
        path_label = self.label_path
        path_raw = self.raw_path

        for img in os.listdir(path_raw + "/image"):

            read_img = cv2.imread(path_raw + "/image/" + img, cv2.IMREAD_GRAYSCALE)
            read_lab = cv2.imread(path_raw + "/label/" + img, cv2.IMREAD_GRAYSCALE)

            # get y and x resolution of image
            y, x = read_img.shape

            # get highest resolution and divide by 2
            size = max(read_img.shape) / 2

            # add 2 to size until size%16 = 0, the final value will determine the size of the 4 sub-images into which the
            # main image will be divided
            while size % 16 != 0:
                size += 2

            final_size = int(size)

            def split_image(image, savedir):

                # upper left corner (pic[y,x])
                u_l = image[0:final_size, 0:final_size]

                # upper right corner
                u_r = image[0:final_size, x - final_size:x]

                # lower left corner
                l_l = image[y - final_size:y, 0:final_size]

                # lower right corner
                l_r = image[y - final_size:y, x - final_size:x]

                cv2.imwrite(savedir + "/" + "0_" + img, u_l)
                cv2.imwrite(savedir + "/" + "1_" + img, u_r)
                cv2.imwrite(savedir + "/" + "2_" + img, l_l)
                cv2.imwrite(savedir + "/" + "3_" + img, l_r)

            split_image(read_img, path_train)
            split_image(read_lab, path_label)

        return final_size


class myAugmentation(object):
    """
    A class used to augment image
    Firstly, read train image and label separately, and then merge them together for the next process
    Secondly, use keras preprocessing to augment image
    Finally, separate augmented image apart into train image and label
    """

    def __init__(self, train_path="data/train/image", label_path="data/train/label", raw_path="data/train/RawImgs",
                 merge_path="data/merge", aug_merge_path="data/aug_merge", aug_train_path="data/aug_train",
                 aug_label_path="data/aug_label", img_type="tif", map_path="data/weights",
                 aug_map_path="data/aug_weights"):

        """
        Using glob to get all .img_type form path
        """

        self.train_imgs = glob.glob(train_path + "/*." + img_type)
        self.label_imgs = glob.glob(label_path + "/*." + img_type)
        self.train_path = train_path
        self.raw_path = raw_path
        self.label_path = label_path
        self.merge_path = merge_path
        self.img_type = img_type
        self.aug_merge_path = aug_merge_path
        self.aug_train_path = aug_train_path
        self.aug_label_path = aug_label_path
        self.slices = len(self.train_imgs)

        self.map_path = map_path
        self.aug_map_path = aug_map_path

        # ImageDataGenerator performs augmentation on original images
        self.datagen = ImageDataGenerator(
            shear_range=0.1,  # 0.005
            horizontal_flip=True,
            vertical_flip=True,
            width_shift_range=0.2,
            height_shift_range=0.2,
            fill_mode='reflect')  # pixels outside boundary are set to 0

    def Augmentation(self, imgnum):

        print("Starting Augmentation \n")

        """
        Start augmentation.....
        """

        trains = self.train_imgs
        labels = self.label_imgs
        path_train = self.train_path
        path_label = self.label_path
        path_merge = self.merge_path
        path_aug_merge = self.aug_merge_path

        # checks if number of files in train and label folder are equal
        if len(trains) != len(labels) or len(trains) == 0 or len(trains) == 0:
            print("trains can't match labels")
            return 0

        # iterate through folder, merge label, original images and save to merged folder
        for count, image in enumerate(os.listdir(path_train)):

            x_t = cv2.imread(path_train + "/" + image, cv2.IMREAD_GRAYSCALE)
            x_l = cv2.imread(path_label + "/" + image, cv2.IMREAD_GRAYSCALE)
            x_w = np.zeros((x_l.shape[0], x_l.shape[1]))

            # create empty array (only 0s) with shape (x,y, number of channels)
            aug_img = np.zeros((x_t.shape[0], x_l.shape[1], 3))

            # setting each channel to label, weights and original
            aug_img[:, :, 2] = x_l
            aug_img[:, :, 1] = x_w
            aug_img[:, :, 0] = x_t

            # write final merged image
            cv2.imwrite(path_merge + "/" + image, aug_img)

            img = aug_img

            img = img.reshape((1,) + img.shape)

            savedir = path_aug_merge + "/" + image

            if not os.path.lexists(savedir):
                os.mkdir(savedir)

            self.doAugmentate(img, savedir, image, imgnum)

    def doAugmentate(self, img, save_to_dir, save_prefix, imgnum, batch_size=1, save_format='tif'):

        """
        augment one image
        """
        datagen = self.datagen
        i = 0
        for batch in datagen.flow(img,
                                  batch_size=batch_size,
                                  save_to_dir=save_to_dir,
                                  save_prefix=save_prefix,
                                  save_format=save_format):

            i += 1

            if i >= imgnum:
                break

    def splitMerge(self):

        print("\nSplitting merged images")

        """
        split merged image apart
        """
        path_merge = self.aug_merge_path
        path_train = self.aug_train_path
        path_label = self.aug_label_path

        for image in os.listdir(path_merge):

            path = path_merge + "/" + image

            train_imgs = glob.glob(path + "/*." + self.img_type)

            savedir = path_train + "/" + image
            if not os.path.lexists(savedir):
                os.mkdir(savedir)

            savedir = path_label + "/" + image
            if not os.path.lexists(savedir):
                os.mkdir(savedir)

            for imgname in train_imgs:
                midname = imgname.split("/")[3]
                img = cv2.imread(imgname)

                # img_train = img[:,:,0]#cv2 read image rgb->bgr
                img_train = img[:, :, 2]  # cv2 read image rgb->bgr
                img_label = img[:, :, 0]

                cv2.imwrite(path_train + "/" + image + "/" + midname, img_train)
                cv2.imwrite(path_label + "/" + image + "/" + midname, img_label)

        print("\n splitMerge finished")


class dataProcess(object):

    def __init__(self, out_rows, out_cols, data_path="data/aug_train", label_path="data/aug_label",
                 test_path="data/test", weight_path="data/aug_weights", npy_path="data/npydata", img_type="tif"):

        """

        """

        self.out_rows = out_rows
        self.out_cols = out_cols
        self.data_path = data_path
        self.label_path = label_path
        self.img_type = img_type
        self.test_path = test_path
        self.npy_path = npy_path
        self.weight_path = weight_path

    def create_train_data(self):

        """
        adding all image data to one numpy array file (npy)

        all mask image files are added to imgs_mask_train.npy
        all original image files are added to imgs_train.npy

        all weight image files are added to weight_train.npy
        """

        i = 0
        print('-' * 30)
        print('Creating training images...')
        print('-' * 30)

        # original
        imgs = glob.glob(self.data_path + "/*/*")

        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        imglabels = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)

        for imgname in imgs:

            midname = imgname.split("/")[2] + "/" + imgname.split("/")[3]

            img = cv2.imread(self.data_path + "/" + midname, cv2.IMREAD_GRAYSCALE)
            label = cv2.imread(self.label_path + "/" + midname, cv2.IMREAD_GRAYSCALE)

            img = np.array([img])
            img = img.reshape((width, height, 1))

            label = np.array([label])
            label = label.reshape((width, height, 1))

            imgdatas[i] = img
            imglabels[i] = label

            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, len(imgs)))
            i += 1

        print('loading done')

        # original
        np.save(self.npy_path + '/imgs_train.npy', imgdatas)
        np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)

        print('Saving to .npy files done.')

    # is used in unet.py script
    def load_train_data(self):

        print('-' * 30)
        print('load train images...')
        print('-' * 30)

        imgs_train = np.load(self.npy_path + "/imgs_train.npy")
        imgs_mask_train = np.load(self.npy_path + "/imgs_mask_train.npy")

        imgs_train = imgs_train.astype('float32')
        imgs_mask_train = imgs_mask_train.astype('float32')

        imgs_train /= 255
        imgs_mask_train /= 255

        imgs_mask_train[imgs_mask_train > 0.5] = 1
        imgs_mask_train[imgs_mask_train <= 0.5] = 0

        return imgs_train, imgs_mask_train

    # is used in unet.py script
    def create_test_data(self):

        """

        adding all image data to one numpy array file (npy)

        all original image files are added to imgs_test.npy

        """

        i = 0
        print('-' * 30)
        print('Creating test images...')
        print('-' * 30)

        print(glob.glob(self.test_path + "/*"))
        imgs = glob.glob(self.test_path + "/*")

        # added 05/12/18 to avoid underscores causing problems when stitching images back together
        if any("_" in s for s in imgs):

            for img in imgs:
                new_img = img.replace("_", "-")
                os.rename(img, new_img)

            imgs = glob.glob(self.test_path + "/*")

        ####
        # this code was added on the 17/01/18 to sort alphanumerical strings

        def atoi(text):
            return int(text) if text.isdigit() else text

        def natural_keys(text):
            return [atoi(c) for c in re.split('(\d+)', text)]

        ####

        imgs.sort(key=natural_keys)

        # create list of images that correspond to arrays in npy file
        ################
        mod_imgs = []
        for x in imgs:

            part = x.split("/")

            c = 0
            while c <= 3:
                mod_imgs.append(part[0] + "/" + part[1] + "/" + str(c) + "_" + part[2])
                c += 1

        ################

        imgdatas = np.ndarray((len(imgs) * 4, self.out_rows, self.out_cols, 1), dtype=np.uint8)
        print(len(imgdatas))

        for imgname in imgs:
            print(imgname)

            # midname = imgname[imgname.rindex("/") + 1:]

            img = cv2.imread(imgname, cv2.IMREAD_GRAYSCALE)

            # insert split into 4 sub-images here
            ################################################### (09/11/18) not tested yet

            y, x = img.shape
            size = max(img.shape) / 2

            while size % 16 != 0:
                size += 2

            final_size = int(size)

            def test_splitter(img_corner, i, final_size=final_size):

                img = img_corner

                img = img.reshape((final_size, final_size, 1))

                imgdatas[i] = img

                i += 1

                return i

            # upper left corner (pic[y,x])
            i = test_splitter(img[0:final_size, 0:final_size], i)
            # upper right corner
            i = test_splitter(img[0:final_size, x - final_size:x], i)
            # lower left corner
            i = test_splitter(img[y - final_size:y, 0:final_size], i)
            # lower right corner
            i = test_splitter(img[y - final_size:y, x - final_size:x], i)

        print('loading done')
        np.save(self.npy_path + '/imgs_test.npy', imgdatas)
        print('Saving to imgs_test.npy files done.')

        return mod_imgs

    # used in unet.py script
    def load_test_data(self):

        print('-' * 30)
        print('load test images...')
        print('-' * 30)

        imgs_test = np.load(self.npy_path + "/imgs_test.npy")
        imgs_test = imgs_test.astype('float32')

        imgs_test /= 255

        return imgs_test


if __name__ == "__main__":
    split = preprocess()

    width = split.splitImgs()
    height = width

    print("New image size is: ", width, "x", height)

    aug = myAugmentation()

    aug.Augmentation(imgnum=80)
    aug.splitMerge()

    mydata = dataProcess(width, height)
    mydata.create_train_data()



