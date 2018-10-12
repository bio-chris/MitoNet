"""

Partially based on
https://github.com/zhixuhao/unet

>>INSERT DESCRIPTION HERE<<

Written for usage on pcd server

Christian Fischer


"""

from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import shutil
import glob
import cv2
import re

# sub-image resolution
width = 656
height = 656

# raw image resolution
org_width = 1300
org_height = 1030


class myAugmentation(object):
    """
    A class used to augment image
    Firstly, read train image and label separately, and then merge them together for the next process
    Secondly, use keras preprocessing to augment image
    Finally, separate augmented image apart into train image and label
    """

    def __init__(self, train_path="data/train/image", label_path="data/train/label", merge_path="data/merge",
                 aug_merge_path="data/aug_merge", aug_train_path="data/aug_train", aug_label_path="data/aug_label",
                 img_type="tif", map_path="data/weights", aug_map_path="data/aug_weights"):

        """
        Using glob to get all .img_type form path
        """

        self.train_imgs = glob.glob(train_path + "/*." + img_type)
        self.label_imgs = glob.glob(label_path + "/*." + img_type)
        self.train_path = train_path
        self.label_path = label_path
        self.merge_path = merge_path
        self.img_type = img_type
        self.aug_merge_path = aug_merge_path
        self.aug_train_path = aug_train_path
        self.aug_label_path = aug_label_path
        self.slices = len(self.train_imgs)

        self.map_path = map_path
        self.aug_map_path = aug_map_path

        self.datagen = ImageDataGenerator(
            shear_range=0.1,  # 0.005
            horizontal_flip=True,
            vertical_flip=True,
            width_shift_range=0.2,
            height_shift_range=0.2,
            fill_mode='reflect')  # pixels outside boundary are set to 0

    def Augmentation(self, weight_map):

        print("Starting Augmentation \n")

        """
        Start augmentation.....
        """
        trains = self.train_imgs
        labels = self.label_imgs
        path_train = self.train_path
        path_label = self.label_path
        path_merge = self.merge_path
        imgtype = self.img_type
        path_aug_merge = self.aug_merge_path
        path_map = self.map_path

        # checks if number of files in train and label folder are equal
        if len(trains) != len(labels) or len(trains) == 0 or len(trains) == 0:
            print("trains can't match labels")
            return 0

        # iterate through folder, merge label, original images and save to merged folder

        for count, image in enumerate(os.listdir(path_train)):

            x_t = cv2.imread(path_train + "/" + image, cv2.IMREAD_GRAYSCALE)
            x_l = cv2.imread(path_label + "/" + image, cv2.IMREAD_GRAYSCALE)

            if weight_map == True:
                pass

            else:

                x_w = np.zeros((x_l.shape[0], x_l.shape[1]))

            # create empty array (only 0s) with shape (x,y, number of channels)
            aug_img = np.zeros((x_t.shape[0], x_l.shape[1], 3))

            # setting each channel to label, weights and original
            aug_img[:, :, 2] = x_l
            aug_img[:, :, 1] = x_w
            aug_img[:, :, 0] = x_t

            # increasing intensity values of label images (to 255 if value was > 0)
            # for x in np.nditer(aug_img[:,:,0], op_flags=['readwrite']):
            #    x[...] = x * 255

            # write final merged image
            cv2.imwrite(path_merge + "/" + image, aug_img)

            img = aug_img

            img = img.reshape((1,) + img.shape)

            savedir = path_aug_merge + "/" + image

            if not os.path.lexists(savedir):
                os.mkdir(savedir)

            self.doAugmentate(img, savedir, image)

    def doAugmentate(self, img, save_to_dir, save_prefix, batch_size=1, save_format='tif', imgnum=80):

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

    def splitMerge(self, weight_map):

        print("\n Splitting merged images")

        """
        split merged image apart
        """
        path_merge = self.aug_merge_path
        path_train = self.aug_train_path
        path_label = self.aug_label_path

        path_map = self.aug_map_path

        for image in os.listdir(path_merge):

            # print(image)

            path = path_merge + "/" + image

            train_imgs = glob.glob(path + "/*." + self.img_type)

            savedir = path_train + "/" + image
            if not os.path.lexists(savedir):
                os.mkdir(savedir)

            savedir = path_label + "/" + image
            if not os.path.lexists(savedir):
                os.mkdir(savedir)

            if weight_map == True:

                savedir = path_map + "/" + image
                if not os.path.lexists(savedir):
                    os.mkdir(savedir)

            for imgname in train_imgs:
                midname = imgname.split("/")[3]
                img = cv2.imread(imgname)

                # img_train = img[:,:,0]#cv2 read image rgb->bgr
                img_train = img[:, :, 2]  # cv2 read image rgb->bgr

                if weight_map == True:
                    img_weight = img[:, :, 1]
                    cv2.imwrite(path_map + "/" + image + "/" + midname, img_weight)

                # img_label = img[:,:,2]
                img_label = img[:, :, 0]

                cv2.imwrite(path_train + "/" + image + "/" + midname, img_train)
                cv2.imwrite(path_label + "/" + image + "/" + midname, img_label)

        print("\n splitMerge finished")


class dataProcess(object):

    def __init__(self, out_rows, out_cols, data_path="data/aug_train", label_path="data/aug_label",
                 test_path="data/test", weight_path="data/aug_weights", npy_path="data/npydata", img_type="tif"):

        self.out_rows = out_rows
        self.out_cols = out_cols
        self.data_path = data_path
        self.label_path = label_path
        self.img_type = img_type
        self.test_path = test_path
        self.npy_path = npy_path
        self.weight_path = weight_path

    def create_train_data(self, weight_map):

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

        # imgs = [x for x in glob.glob(self.data_path+"/*/*") if image not in x]

        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        imglabels = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)

        if weight_map == True:
            imgweights = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)

        for imgname in imgs:

            midname = imgname.split("/")[2] + "/" + imgname.split("/")[3]

            # print(midname)
            # print(self.data_path + "/" + midname)

            img = cv2.imread(self.data_path + "/" + midname, cv2.IMREAD_GRAYSCALE)
            label = cv2.imread(self.label_path + "/" + midname, cv2.IMREAD_GRAYSCALE)

            if weight_map == True:
                weights = cv2.imread(self.weight_path + "/" + midname, cv2.IMREAD_GRAYSCALE)

                weights = np.array([weights])
                weights = weights.reshape((width, height, 1))

                imgweights[i] = weights

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

        if weight_map == True:
            np.save(self.npy_path + '/imgs_weights_train.npy', imgweights)

        print('Saving to .npy files done.')

    """
    17/09/18

    ADD SECTION TO DELETE ALL FILES GENERATED IN AUG-LABEL, AUG-MERGE, AUG-TRAIN, AUG-WEIGHTS AND MERGE 
    AFTER SUCCESSFUL COMPLETION OF CREATE TRAIN DATA 
    """

    def del_data(self):

        aug_label_file = os.listdir("data/aug_label")

        train_data = os.listdir("data/train/image")

        # deleting augmented data
        for files in aug_label_file:
            shutil.rmtree("data/aug_label/" + files)
            shutil.rmtree("data/aug_merge/" + files)
            shutil.rmtree("data/aug_train/" + files)

            os.remove("data/merge/" + files)

        # deleting train data
        for files in train_data:
            os.remove("data/train/image/" + files)
            os.remove("data/train/label/" + files)

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

        # return imgs_train, imgs_mask_train, imgs_weight_train

        # use if no weights should be loaded
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

        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)

        ####
        # this code was added on the 17/01/18 to sort alphanumerical strings

        def atoi(text):
            return int(text) if text.isdigit() else text

        def natural_keys(text):
            return [atoi(c) for c in re.split('(\d+)', text)]

        ####

        imgs.sort(key=natural_keys)

        for imgname in imgs:
            print(imgname)

            midname = imgname[imgname.rindex("/") + 1:]

            img = cv2.imread(imgname, cv2.IMREAD_GRAYSCALE)
            img = np.array([img])

            img = img.reshape((width, height, 1))

            # img = img.reshape((org_height,org_width,1))

            imgdatas[i] = img
            i += 1

        print('loading done')
        np.save(self.npy_path + '/imgs_test.npy', imgdatas)
        print('Saving to imgs_test.npy files done.')

        return imgs

    # used in unet.py script
    def load_test_data(self):

        print('-' * 30)
        print('load test images...')
        print('-' * 30)

        imgs_test = np.load(self.npy_path + "/imgs_test.npy")
        imgs_test = imgs_test.astype('float32')

        imgs_test /= 255

        return imgs_test


"""
def create_weight_map(label, w0=10, sigma=5):

    # creating first parameter of weight map formula
    class_weight_map = np.ones_like(label)
    class_weight_map[label > 0] = 2

    # setting all 255 values to 1
    label[label > 1] = 1
    # inverting label for distance_transform
    new_label = 1-label

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

                    dist_map2[x,y] = img[x,y]
                    break

                else:
                    dist_map2[x,y] = dist_map1[x,y]

            y+=1

        x+=1

    weight_map = class_weight_map + w0 * np.exp(- ((dist_map1+dist_map2)**2 / (2 * sigma ** 2)))

    return weight_map
"""


# 25/09/18: additional functions to generate all npy files in one run

def move_train_files(file):
    dest_image = "data/train/image"
    dest_label = "data/train/label"

    label_files = os.listdir("data/train/CV_2/New_Label/" + file)

    for label in label_files:

        full_label_name = "data/train/CV_2/New_Label/" + file + "/" + label
        full_train_name = "data/train/CV_2/New_Train/" + file + "/" + label

        if os.path.isfile(full_label_name):
            shutil.copy(full_label_name, dest_label)
            shutil.copy(full_train_name, dest_image)


def move_npy_files(file):
    os.makedirs("data/npydata/" + file)

    shutil.move("data/npydata/imgs_mask_train.npy", "data/npydata/" + file + "/imgs_mask_train.npy")
    shutil.move("data/npydata/imgs_train.npy", "data/npydata/" + file + "/imgs_train.npy")


if __name__ == "__main__":
    """
    aug = myAugmentation()

    # insert True, when weight map should be generated, else insert False 
    weight_map = False

    aug.Augmentation(weight_map)
    aug.splitMerge(weight_map)

    mydata = dataProcess(width,height)

    mydata.create_train_data(weight_map)
    mydata.del_data()
    """

    """
    # used for cross validation, automated generation of npy data

    file_list = os.listdir("data/train/CV_2/New_Label")

    # count = 0


    for file in file_list:
        print(file)

        # if count == 3:
        #    break

        # moving files from CV2 folder to respective label and image folders
        move_train_files(file)

        # time.sleep(20)

        aug = myAugmentation()
        mydata = dataProcess(width, height)

        # data augmentation
        aug.Augmentation()
        aug.splitMerge()

        # creating npy files for unet
        mydata.create_train_data()

        # moving npy files into separate folders
        move_npy_files(file)

        # clear data for next run
        mydata.del_data()

        # count += 1
    """




