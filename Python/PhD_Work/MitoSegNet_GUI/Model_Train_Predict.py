""""

Partially based on
https://github.com/zhixuhao/unet

>>INSERT DESCRIPTION HERE<<


Written for usage on pcd server (linux)

Christian Fischer

---

Notes:

weight map functionality has been removed (24/09/18): accuracy does not improve with usage of weigth map


number of conv layers: 24
number of relu units: 23
number of sigmoid units: 1 (after last conv layer)
number of batch norm layers: 10
number of max pooling layers: 4


"""


import os
import cv2
import re
import numpy as np
import copy
import glob
from math import sqrt
from skimage.morphology import remove_small_objects
from scipy.ndimage import label



class GPU_or_CPU(object):


    def __init__(self, mode):

        self.mode = mode

    def ret_mode(self):

        if self.mode == "GPU":
            print("Train / Predict on GPU")

        elif self.mode == "CPU":
            print("Train / Predict on CPU")
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        return self.mode


from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Activation, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.initializers import RandomNormal as gauss
from keras import backend as K
from keras import losses
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.preprocessing.image import array_to_img

from Training_DataGenerator import *


"""

16/01/18

Image size needs to be divisible by 2:
To allow seamless tiling of the output segmentation map, input tile size needs to be selected such that all 2x2
max pooling operations are applied to a layer with an even x and y size

"""


class MitoSegNet(object):


    def __init__(self, path, img_rows, img_cols, org_img_rows, org_img_cols):

        self.path = path

        self.img_rows = img_rows
        self.img_cols = img_cols

        self.org_img_rows = org_img_rows
        self.org_img_cols = org_img_cols



    def load_data(self, wmap, vbal):


        print('-' * 30)
        print('Load train images...')
        print('-' * 30)

        imgs_train = np.load(self.path + os.sep + "npydata" + os.sep +"imgs_train.npy")
        imgs_mask_train = np.load(self.path + os.sep + "npydata" + os.sep + "imgs_mask_train.npy")

        imgs_train = imgs_train.astype('float32')
        imgs_mask_train = imgs_mask_train.astype('float32')

        imgs_train /= 255
        imgs_mask_train /= 255

        imgs_mask_train[imgs_mask_train > 0.5] = 1
        imgs_mask_train[imgs_mask_train <= 0.5] = 0

        if wmap == True:

            imgs_weights = np.load(self.path + os.sep + "npydata" + os.sep + "imgs_weights.npy")
            imgs_weights = imgs_weights.astype('float32')

            # setting background pixel weights to vbal (because of class imbalance)
            imgs_weights[imgs_weights == 1] = vbal

            return imgs_train, imgs_mask_train, imgs_weights

        else:

            return imgs_train, imgs_mask_train


    def get_mitosegnet(self, wmap, lr):

        inputs = Input(shape=(self.img_rows, self.img_cols, 1))
        print(inputs.get_shape(), type(inputs))


        # core mitosegnet (modified u-net) architecture
        ######################################
        ######################################

        """

        as of 10/10/2018

        Contracting path:

            5 sections each consisting of the following layers:

                convolution 
                batchnorm 
                activation 
                convolution
                batchnorm
                activation 
                pooling

        Expanding path: 

            4 sections each consisting of the following layers:

                inverse convolution (upsampling)
                merging
                convolution
                convolution 

            1 section containing only one convolutional layer 

        """


        # batchnorm architecture (batchnorm before activation)
        ######################################################################
        #"""


        conv1 = Conv2D(64, 3, padding='same', kernel_initializer=gauss())(inputs)
        print("conv1 shape:", conv1.shape)
        batch1 = BatchNormalization()(conv1)
        act1 = Activation("relu")(batch1)

        conv1 = Conv2D(64, 3, padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 64))))(act1)  # conv1
        print("conv1 shape:", conv1.shape)
        batch1 = BatchNormalization()(conv1)
        act1 = Activation("relu")(batch1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(act1)
        print("pool1 shape:", pool1.shape)
        ########

        ########
        conv2 = Conv2D(128, 3, padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 64))))(pool1)
        print("conv2 shape:", conv2.shape)
        batch2 = BatchNormalization()(conv2)
        act2 = Activation("relu")(batch2)

        conv2 = Conv2D(128, 3, padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 128))))(act2)  # conv2
        print("conv2 shape:", conv2.shape)
        batch2 = BatchNormalization()(conv2)
        act2 = Activation("relu")(batch2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(act2)
        print("pool2 shape:", pool2.shape)
        ########

        ########
        conv3 = Conv2D(256, 3, padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 128))))(pool2)
        print("conv3 shape:", conv3.shape)
        batch3 = BatchNormalization()(conv3)
        act3 = Activation("relu")(batch3)

        conv3 = Conv2D(256, 3, padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 256))))(act3)  # conv3
        print("conv3 shape:", conv3.shape)
        batch3 = BatchNormalization()(conv3)
        act3 = Activation("relu")(batch3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(act3)
        print("pool3 shape:", pool3.shape)
        ########

        ########
        conv4 = Conv2D(512, 3, padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 256))))(pool3)
        batch4 = BatchNormalization()(conv4)
        act4 = Activation("relu")(batch4)

        conv4 = Conv2D(512, 3, padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 512))))(act4)  # conv4
        batch4 = BatchNormalization()(conv4)
        act4 = Activation("relu")(batch4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(act4)
        ########

        ########
        conv5 = Conv2D(1024, 3, padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 512))))(pool4)
        batch5 = BatchNormalization()(conv5)
        act5 = Activation("relu")(batch5)

        conv5 = Conv2D(1024, 3, padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 1024))))(act5)  # conv5
        batch5 = BatchNormalization()(conv5)
        act5 = Activation("relu")(batch5)
        ########

        up6 = Conv2D(512, 2, activation='relu', padding='same',
                     kernel_initializer=gauss(stddev=sqrt(2 / (9 * 1024))))(UpSampling2D(size=(2, 2))(act5))

        merge6 = concatenate([conv4, up6], axis=3)
        #merge6 = merge([act4, up6], mode='concat', concat_axis=3)


        conv6 = Conv2D(512, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 512))))(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 512))))(conv6)


        up7 = Conv2D(256, 2, activation='relu', padding='same',
                     kernel_initializer=gauss(stddev=sqrt(2 / (9 * 512))))(UpSampling2D(size=(2, 2))(conv6))

        merge7 = concatenate([conv3, up7], axis=3)
        #merge7 = merge([act3, up7], mode='concat', concat_axis=3)

        conv7 = Conv2D(256, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 256))))(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 256))))(conv7)


        up8 = Conv2D(128, 2, activation='relu', padding='same',
                     kernel_initializer=gauss(stddev=sqrt(2 / (9 * 256))))(UpSampling2D(size=(2, 2))(conv7))

        merge8 = concatenate([conv2, up8], axis=3)
        #merge8 = merge([act2, up8], mode='concat', concat_axis=3)

        conv8 = Conv2D(128, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 128))))(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 128))))(conv8)


        up9 = Conv2D(64, 2, activation='relu', padding='same',
                     kernel_initializer=gauss(stddev=sqrt(2 / (9 * 128))))(UpSampling2D(size=(2, 2))(conv8))

        merge9 = concatenate([conv1, up9], axis=3)
        #merge9 = merge([act1, up9], mode='concat', concat_axis=3)

        conv9 = Conv2D(64, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 64))))(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 64))))(conv9)

        conv9 = Conv2D(2, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 64))))(conv9)

        #"""
        ######################################################################

        conv10 = Conv2D(1, 1, activation='sigmoid', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 2))))(conv9)


        if wmap == False:
            input = inputs
            loss = self.pixelwise_crossentropy()
        else:
            weights = Input(shape=(self.img_rows, self.img_cols, 1))
            input = [inputs, weights]

            loss = self.weighted_pixelwise_crossentropy(input[1])
            #loss = self.alternative_loss(input[1])


        model = Model(inputs=input, outputs=conv10)

        # normally set to 1e-4
        model.compile(optimizer=Adam(lr=lr), loss=loss, metrics=['accuracy', self.dice_coefficient])


        return model


    def dice_coefficient(self, y_true, y_pred):

        smooth = 1

        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)

        intersection = K.sum(y_true_f * y_pred_f)

        dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

        return dice

    def pixelwise_crossentropy(self):

        def loss(y_true, y_pred):

            return losses.binary_crossentropy(y_true, y_pred)

        return loss

    def weighted_pixelwise_crossentropy(self, wmap):

        def loss(y_true, y_pred):

            return losses.binary_crossentropy(y_true, y_pred) * wmap

        return loss


    def train(self, epochs, learning_rate, batch_size, wmap, vbal, model_name):

        print("Loading data")

        if wmap == False:
            imgs_train, imgs_mask_train = self.load_data(wmap=wmap, vbal=vbal)
        else:
            imgs_train, imgs_mask_train, img_weights = self.load_data(wmap=wmap, vbal=vbal)

        print("Loading data done")

        model = self.get_mitosegnet(wmap, learning_rate)
        print("Got mitosegnet")


        print(self.path + os.sep + model_name)


        if os.path.isfile(self.path + os.sep + model_name):

            model.load_weights(self.path + os.sep + model_name)
            print("Loading weights")

        else:
            print("No previously optimized weights were loaded. Proceeding without")


        # Set network weights saving mode.
        # save previously established network weights (saving model after every epoch)

        print('Fitting model...')

        csv_logger = CSVLogger(self.path + os.sep + model_name + '_training_log.csv')

        # Set callback functions to early stop training and save the best model so far
        callbacks = [EarlyStopping(monitor='val_loss', patience=5),
                     ModelCheckpoint(filepath=self.path + os.sep +model_name+".hdf5", monitor='val_loss', verbose=1, save_best_only=True),
                     csv_logger]


        if wmap == True:
            x = [imgs_train, img_weights]
        else:
            x = imgs_train

        model.fit(x=x, y=imgs_mask_train, batch_size=batch_size, epochs=epochs, verbose=1,
                            validation_split=0.2, shuffle=True, callbacks=callbacks)


        info_file = open(self.path + os.sep + model_name + "_train_info.txt", "w")
        info_file.write("Learning rate: " + str(learning_rate)+
                        "\nBatch size: " + str(batch_size))
        info_file.close()

        K.clear_session()


    def predict(self, test_path, wmap, tile_size, n_tiles, model_name, pretrain):

        """
        :return:
        """

        org_img_rows = self.org_img_rows
        org_img_cols = self.org_img_cols


        def create_test_data(tile_size, n_tiles):

            #
            # adding all image data to one numpy array file (npy)

            # all original image files are added to imgs_test.npy
            #


            i = 0
            print('-' * 30)
            print('Creating test images...')
            print('-' * 30)

            imgs = glob.glob(test_path + os.sep + "*")


            # added 05/12/18 to avoid underscores causing problems when stitching images back together
            #if any("_" in s for s in imgs):

            for img in imgs:

                if "_" in img and ".tif" in img:

                    # split file path without filename
                    img_edited = img.split(os.sep)[:-1]

                    # join list back to path string
                    img_edited_path = os.sep.join(img_edited)

                    img_name = img.split(os.sep)[-1]
                    img_name = img_name.replace("_", "-")

                    os.rename(img, img_edited_path + os.sep + img_name)

            imgs = glob.glob(test_path + os.sep + "*")

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

                part = x.split(os.sep)

                c = 0
                while c <= n_tiles - 1:

                    temp_str = os.sep.join(part[:-1])

                    if ".tif" in part[-1]:

                        mod_imgs.append(temp_str + os.sep + str(c) + "_" + part[-1])

                    c += 1

            ################

            imgdatas = np.ndarray((len(imgs) * n_tiles, tile_size, tile_size, 1), dtype=np.uint8)

            for imgname in imgs:

                if ".tif" in imgname:

                    print(imgname)

                    img = cv2.imread(imgname, cv2.IMREAD_GRAYSCALE)
                    cop_img = copy.copy(img)

                    # insert split into 4 sub-images here
                    ################################################### (09/11/18)

                    y, x = img.shape


                    def small_img_size(y, x, img):

                        height = y
                        width = x

                        org_height, org_width = 1030, 1300

                        img_template = np.zeros((org_height, org_width))

                        y_pos = 0
                        while y_pos < 1030:

                            new_y_pos = y_pos + height

                            if new_y_pos >= 1030:
                                new_y_pos = 1030
                                final_y = new_y_pos - y_pos

                            else:
                                final_y = height

                            x_pos = 0
                            while x_pos < 1300:

                                new_x_pos = x_pos + width

                                if new_x_pos >= 1300:
                                    new_x_pos = 1300
                                    img_template[y_pos:new_y_pos, x_pos:new_x_pos] = img[:final_y, :new_x_pos - x_pos]

                                if new_x_pos < 1300:
                                    img_template[y_pos:new_y_pos, x_pos:new_x_pos] = img[:final_y, :width]

                                x_pos = new_x_pos

                            y_pos = new_y_pos

                        img = img_template
                        cop_img = copy.copy(img)

                        y, x = img.shape

                        return img, cop_img, y, x


                    if y < 656 or x < 656:

                        img, cop_img, y, x = small_img_size(y,x, img)


                    start_y = 0
                    start_x = 0
                    end_y = tile_size
                    end_x = tile_size

                    column = 0
                    row = 0

                    for n in range(n_tiles):
                        start_x, end_x, start_y, end_y, column, row = preproc.find_tile_pos(x, y, tile_size, start_x, end_x,
                                                                                         start_y, end_y, column, row)

                        img_tile = cop_img[start_y:end_y, start_x:end_x]

                        img = img_tile.reshape((tile_size, tile_size, 1))

                        imgdatas[i] = img

                        i += 1

                np.save(test_path + os.sep + 'imgs_array.npy', imgdatas)

            return mod_imgs

        def load_test_data():

            print('-' * 30)
            print('Load test images...')
            print('-' * 30)

            imgs_test = np.load(test_path + os.sep + "imgs_array.npy")
            imgs_test = imgs_test.astype('float32')

            imgs_test /= 255

            return imgs_test

        preproc = Preprocess()

        l_imgs = create_test_data(int(tile_size), int(n_tiles))
        imgs_test = load_test_data()

        # predict if no npy array exists yet
        if not os.path.isfile(test_path + os.sep + "imgs_mask_array.npy"):

            lr = 1e-4

            model = self.get_mitosegnet(wmap, lr)

            if pretrain == "":
                model.load_weights(self.path + os.sep + model_name)
            else:
                model.load_weights(pretrain)

            print('Predict test data')

            # batch size of 14 was too large to fit into GPU memory
            # verbose: show ongoing progress (0 do not show)
            imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)

            np.save(test_path + os.sep + 'imgs_mask_array.npy', imgs_mask_test)

        else:

            print("\nFound imgs_mask_array.npy. Skipping prediction and converting array to images\n")

        org_img_list = []

        for img in l_imgs:

            # split file path without filename

            if ".tif" in img:

                #img_edited = img.split(os.sep)[:-1]
                # join list back to path string
                #img_edited_path = os.sep.join(img_edited)

                img_name = img.split(os.sep)[-1]
                img_name = img_name.split("_")[1]

                org_img_list.append(img_name)

        org_img_list = list(set(org_img_list))


        def atoi(text):
            return int(text) if text.isdigit() else text

        def natural_keys(text):
            return [atoi(c) for c in re.split('(\d+)', text)]

        #alphanumeric sorting of list elements
        org_img_list.sort(key=natural_keys)

        # saving arrays as images
        print("Array to image")

        imgs = np.load(test_path + os.sep + 'imgs_mask_array.npy')

        #####
        imgs = imgs.astype('float32')
        #####

        start_y = 0
        start_x = 0
        end_y = tile_size
        end_x = tile_size

        column = 0
        row = 0

        img_nr = 0
        org_img_list_index = 0


        small_size = False
        for n, image in zip(range(imgs.shape[0]), l_imgs):

            if img_nr == 0:

                if org_img_rows < 656 or org_img_cols < 656:

                    small_size = True

                    small_y, small_x = org_img_rows, org_img_cols
                    org_img_rows, org_img_cols = 1030, 1300

                current_img = np.zeros((org_img_rows, org_img_cols))


            img = imgs[n]

            # setting any prediction with a probability of 0.5 or lower to 0 and above 0.5 to 1 (255 in 8 bit)
            img[img > 0.5] = 1
            img[img <= 0.5] = 0

            img = array_to_img(img)

            start_x, end_x, start_y, end_y, column, row = preproc.find_tile_pos(org_img_cols, org_img_rows, tile_size,
                                                                   start_x, end_x, start_y, end_y, column, row)

            current_img[start_y:end_y, start_x:end_x] = img


            img_nr += 1

            # once one image has been fully stitched, remove any objects below 10 px size and save
            if img_nr == n_tiles:

                start_y = 0
                start_x = 0
                end_y = tile_size
                end_x = tile_size

                column = 0
                row = 0

                label_image, num_features = label(current_img)
                new_image = remove_small_objects(label_image, 10)

                new_image[new_image != 0] = 255

                if small_size == True:

                    new_image = new_image[:small_y, :small_x]

                cv2.imwrite(test_path + os.sep + "Prediction" + os.sep + org_img_list[org_img_list_index], new_image)

                org_img_list_index+=1
                img_nr = 0

        K.clear_session()



