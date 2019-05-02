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
import math
import os
import copy
import glob
import cv2
import re
from keras.preprocessing.image import ImageDataGenerator
from skimage.measure import label as set_labels, regionprops
from scipy.ndimage.morphology import distance_transform_edt as get_dmap


class Preprocess(object):

    def __init__(self, train_path="data/train/image", label_path="data/train/label", raw_path = "data/train/RawImgs",
                 img_type="tif"):

        """
        Using glob to get all .img_type form path
        """

        self.train_imgs = glob.glob(train_path + "/*." + img_type)
        self.label_imgs = glob.glob(label_path + "/*." + img_type)
        self.train_path = train_path
        self.raw_path = raw_path
        self.label_path = label_path


    def poss_tile_sizes(self):

        """
        get corresponding tile sizes and number of tiles per raw image
        """

        path_raw = self.raw_path

        for img in os.listdir(path_raw + "/image"):
            read_img = cv2.imread(path_raw + "/image/" + img, -1)
            y,x = read_img.shape

            break

        size = 16

        while size < max([y, x]) / 2 + 16:

            x_tile = math.ceil(x / size)
            y_tile = math.ceil(y / size)

            x_overlap = (np.abs(x - x_tile * size)) / (x_tile - 1)
            y_overlap = (np.abs(y - y_tile * size)) / (y_tile - 1)

            if (x_overlap.is_integer() and y_overlap.is_integer()) and (x_tile * y_tile) % 2 == 0:
                print("tile size (px):", size, "number of tiles: ", x_tile * y_tile)

            size += 16




    def find_tile_pos(self, x, y, tile_size, start_x, end_x, start_y, end_y, column, row):

        """
        :param x:
        :param y:
        :param tile_size:
        :param start_x:
        :param end_x:
        :param start_y:
        :param end_y:
        :param column:
        :param row:
        :return: start x, end x, start y and end y coordinates for tile position
        """

        x_tile = math.ceil(x / tile_size)
        y_tile = math.ceil(y / tile_size)

        x_overlap = (np.abs(x - x_tile * tile_size)) / (x_tile - 1)
        y_overlap = (np.abs(y - y_tile * tile_size)) / (y_tile - 1)

        # if column greater equal 1 then set start_x and end_x as follows
        if column >= 1:
            start_x = int(column * tile_size - column * x_overlap)
            end_x = int(start_x + tile_size)

        # if row greater equal 1 then set start_y and end_y as follows
        if row >= 1:
            start_y = int((row) * tile_size - (row) * y_overlap)
            end_y = int(start_y + tile_size)

        # if column is equal to number of x tiles, reset start_x, end_x and column (moving to next row)
        if column == x_tile:
            start_x = 0
            end_x = tile_size

            column = 0

        # if column greater equal number of x tiles -1, add 1 to row (moving to next column)
        if column >= x_tile - 1 and row < y_tile - 1:
            row += 1

        column += 1


        return start_x, end_x, start_y, end_y, column, row


    def splitImgs(self, tile_size, n_tiles):

        """
        split original images into n tiles of equal width and length

        :param tile_size:
        :param n_tiles:
        :return:
        """

        if n_tiles%2!=0 or tile_size%16!=0:
            print("Incorrect number of tiles or tile size not divisible by 16.\nAborting")
            exit()


        path_train = self.train_path
        path_label = self.label_path
        path_raw = self.raw_path

        for img in os.listdir(path_raw + "/image"):


            read_img = cv2.imread(path_raw + "/image/" + img, -1)

            if np.sum(read_img) == 0:
                print("Problem with reading image.\nAborting")
                exit()

            elif np.max(read_img) > 255:
                print("Image bit depth is 16 or higher. Please convert images to 8-bit first.\nAborting")
                exit()

            read_lab = cv2.imread(path_raw + "/label/" + img, cv2.IMREAD_GRAYSCALE)
            y, x = read_img.shape

            if tile_size > max(y,x)/2+16:
                print("Tile size to big.\nAborting")
                exit()


            # splitting image into n tiles of predefined size
            #############

            start_y = 0
            start_x = 0
            end_y = tile_size
            end_x = tile_size

            column = 0
            row = 0

            for i in range(n_tiles):

                start_x, end_x, start_y, end_y, column, row = self.find_tile_pos(x, y, tile_size, start_x, end_x, start_y, end_y,
                                                                    column, row)

                image_tile_train = read_img[start_y:end_y, start_x:end_x]
                image_tile_label = read_lab[start_y:end_y, start_x:end_x]

                cv2.imwrite(path_train + "/" + str(i) + "_" + img, image_tile_train)
                cv2.imwrite(path_label + "/" + str(i) + "_" + img, image_tile_label)

                #############


class Augment(object):
    """
    A class used to augment image
    Firstly, read train image and label separately, and then merge them together for the next process
    Secondly, use keras preprocessing to augment image
    Finally, separate augmented image apart into train image and label
    """

    def __init__(self, train_path="data/train/image", label_path="data/train/label", raw_path = "data/train/RawImgs",
                 merge_path="data/merge", aug_merge_path="data/aug_merge", aug_train_path="data/aug_train",
                 aug_label_path="data/aug_label", img_type="tif",
                 weights_path="data/weights", aug_weights_path="data/aug_weights"):

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
        self.aug_weights_path = aug_weights_path
        self.aug_label_path = aug_label_path
        self.slices = len(self.train_imgs)

        self.map_path = weights_path
        #self.aug_map_path = aug_map_path

        # ImageDataGenerator performs augmentation on original images
        self.datagen = ImageDataGenerator(

            shear_range=0.2,  # originally set to 0.1
            rotation_range=180,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            width_shift_range=0.2,
            height_shift_range=0.2,
            fill_mode='reflect')  # pixels outside boundary are set to 0



    def start_augmentation(self, imgnum, wmap):


        def create_distance_weight_map(label, w0=10, sigma=5):

            # creating first parameter of weight map formula
            class_weight_map = np.ones_like(label)
            class_weight_map[label > 0] = 2

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

            weight_map = class_weight_map + w0 * np.exp(- ((dist_map1 + dist_map2) ** 2 / (2 * sigma ** 2)))

            return weight_map


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
            print("Number of train images does match number of label images.\nAborting")
            exit()

        # iterate through folder, merge label, original images and save to merged folder
        for count, image in enumerate(os.listdir(path_train)):

            print(image)

            x_t = cv2.imread(path_train + "/" + image, cv2.IMREAD_GRAYSCALE)
            x_l = cv2.imread(path_label + "/" + image, cv2.IMREAD_GRAYSCALE)


            if wmap == False:
                x_w = np.zeros((x_l.shape[0], x_l.shape[1]))

            else:
                #create weight map
                x_w = create_distance_weight_map(x_l)

            # create empty array (only 0s) with shape (x,y, number of channels)
            aug_img = np.zeros((x_t.shape[0], x_l.shape[1], 3))

            # setting each channel to label, empty array and original

            aug_img[:, :, 2] = x_l
            aug_img[:, :, 1] = x_w
            aug_img[:, :, 0] = x_t

            if wmap == True:

                #aug_img[:, :, 2][aug_img[:, :, 2]>1] = 255
                #aug_img[:, :, 2][aug_img[:, :, 2] <= 1] = 0

                #increasing intensity values of label images (to 255 if value was > 0)
                for x in np.nditer(aug_img[:,:,2], op_flags=['readwrite']):
                    x[...] = x * 255


            # write final merged image
            cv2.imwrite(path_merge + "/" + image, aug_img)

            img = aug_img
            img = img.reshape((1,) + img.shape)

            savedir = path_aug_merge + "/" + image

            if not os.path.lexists(savedir):
                os.mkdir(savedir)

            self.doAugmentate(img, savedir, image, imgnum)


    def doAugmentate(self, img, save_to_dir, save_prefix, imgnum , batch_size=1, save_format='tif'):

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

    def splitMerge(self, wmap):

        print("Splitting merged images")

        """
        split merged image apart
        """
        path_merge = self.aug_merge_path
        path_train = self.aug_train_path
        path_weights = self.aug_weights_path
        path_label = self.aug_label_path


        for image in os.listdir(path_merge):

            path = path_merge + "/" + image

            train_imgs = glob.glob(path + "/*." + self.img_type)


            def save_dir(path):
                savedir = path + "/" + image
                if not os.path.lexists(savedir):
                    os.mkdir(savedir)

            save_dir(path_train)
            save_dir(path_label)

            """
            savedir = path_train + "/" + image
            if not os.path.lexists(savedir):
                os.mkdir(savedir)

            savedir = path_label + "/" + image
            if not os.path.lexists(savedir):
                os.mkdir(savedir)
            """

            if wmap == True:
                save_dir(path_weights)


            for imgname in train_imgs:
                midname = imgname.split("/")[3]
                img = cv2.imread(imgname)

                img_train = img[:, :, 2]  # cv2 read image rgb->bgr
                img_label = img[:, :, 0]

                cv2.imwrite(path_train + "/" + image + "/" + midname, img_train)
                cv2.imwrite(path_label + "/" + image + "/" + midname, img_label)

                if wmap==True:
                    img_weights = img[:, :, 1]
                    cv2.imwrite(path_weights + "/" + image + "/" + midname, img_weights)

        print("\nsplitMerge finished")


class Create_npy_files(Preprocess):

    def __init__(self, out_rows, out_cols, data_path="data/aug_train", label_path="data/aug_label",
                 test_path="data/test", weight_path="data/aug_weights", npy_path="data/npydata", img_type="tif"):



        Preprocess.__init__(self, train_path="data/train/image", label_path="data/train/label",
                            raw_path = "data/train/RawImgs", img_type=img_type)

        self.out_rows = out_rows
        self.out_cols = out_cols
        self.data_path = data_path
        self.label_path = label_path
        self.img_type = img_type
        self.test_path = test_path
        self.npy_path = npy_path
        self.weight_path = weight_path


    def create_train_data(self, wmap):

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
        imgweights = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)


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

            if wmap==True:

                weights = cv2.imread(self.weight_path + "/" + midname,cv2.IMREAD_GRAYSCALE)

                weights = np.array([weights])
                weights = weights.reshape((width, height, 1))

                imgweights[i] = weights


            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, len(imgs)))
            i += 1

        print('Loading done')

        # original
        np.save(self.npy_path + '/imgs_train.npy', imgdatas)
        np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)

        if wmap==True:
            np.save(self.npy_path + '/imgs_weights.npy', imgweights)

        print('Saving to .npy files done.')

    def check_class_balance(self):

        label_array = np.load(self.npy_path + "/imgs_mask_train.npy")

        tile_size = label_array[0].shape[0]

        l = []
        for count, i in enumerate(label_array):

            b = len(i[i == 0])
            l.append(b / (tile_size ** 2))

        av = np.average(l)

        print("Average percentage of 0 pixels in label array:", av)
        print("Foreground to background pixel ratio:", 1, "to", math.ceil(1/(1-av)))


    # is used in MitoSegNet script
    def load_train_data(self, wmap, vbal):

        print('-' * 30)
        print('Load train images...')
        print('-' * 30)

        imgs_train = np.load(self.npy_path + "/imgs_train.npy")
        imgs_mask_train = np.load(self.npy_path + "/imgs_mask_train.npy")

        imgs_train = imgs_train.astype('float32')
        imgs_mask_train = imgs_mask_train.astype('float32')

        imgs_train /= 255
        imgs_mask_train /= 255

        imgs_mask_train[imgs_mask_train > 0.5] = 1
        imgs_mask_train[imgs_mask_train <= 0.5] = 0

        if wmap==True:

            imgs_weights = np.load(self.npy_path + "/imgs_weights.npy")
            imgs_weights = imgs_weights.astype('float32')

            # setting background pixel weights to vbal (because of class imbalance)
            imgs_weights[imgs_weights == 1] = vbal

            return imgs_train, imgs_mask_train, imgs_weights

        else:

            return imgs_train, imgs_mask_train


    # is used in MitoSegNet script
    def create_test_data(self, tile_size, n_tiles):

        """
        adding all image data to one numpy array file (npy)

        all original image files are added to imgs_test.npy
        """

        i = 0
        print('-' * 30)
        print('Creating test images...')
        print('-' * 30)

        #print(glob.glob(self.test_path + "/*"))
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
            while c <= n_tiles-1:
                mod_imgs.append(part[0] + "/" + part[1] + "/" + str(c) + "_" + part[2])
                c += 1

        ################

        imgdatas = np.ndarray((len(imgs) * n_tiles, self.out_rows, self.out_cols, 1), dtype=np.uint8)

        for imgname in imgs:
            print(imgname)

            img = cv2.imread(imgname, cv2.IMREAD_GRAYSCALE)
            cop_img = copy.copy(img)

            # insert split into 4 sub-images here
            ################################################### (09/11/18)

            y, x = img.shape

            start_y = 0
            start_x = 0
            end_y = tile_size
            end_x = tile_size

            column = 0
            row = 0

            for n in range(n_tiles):

                start_x, end_x, start_y, end_y, column, row = self.find_tile_pos(x, y, tile_size, start_x, end_x,
                                                                                 start_y, end_y, column, row)

                img_tile = cop_img[start_y:end_y, start_x:end_x]

                img = img_tile.reshape((tile_size, tile_size, 1))

                imgdatas[i] = img

                i+=1


        print('Loading done')
        np.save(self.npy_path + '/imgs_test.npy', imgdatas)
        print('Saving to imgs_test.npy files done.')

        return mod_imgs


    # used in MitoSegNet script
    def load_test_data(self):

        print('-' * 30)
        print('Load test images...')
        print('-' * 30)

        imgs_test = np.load(self.npy_path + "/imgs_test.npy")
        imgs_test = imgs_test.astype('float32')

        imgs_test /= 255

        return imgs_test


if __name__ == "__main__":

    preproc = Preprocess()

    #preproc.poss_tile_sizes()

    n_tiles = 6
    width = 576
    height = width

    preproc.splitImgs(width, n_tiles)

    wmap = True

    aug = Augment()
    #aug.start_augmentation(imgnum=4, wmap=wmap)
    #aug.splitMerge(wmap=wmap)

    mydata = Create_npy_files(width, height)

    #mydata.create_train_data(wmap=wmap)
    #mydata.check_class_balance()



