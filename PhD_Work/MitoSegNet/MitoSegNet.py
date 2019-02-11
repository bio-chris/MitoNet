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

import pandas as pd
from keras.models import *
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Activation, BatchNormalization, Dropout
from keras.optimizers import *
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.initializers import RandomNormal as gauss
from keras.preprocessing.image import array_to_img
from keras import losses
from Training_DataGenerator import *
import cv2
import os
import re
from math import sqrt
from skimage.morphology import remove_small_objects
from scipy.ndimage import label

"""

16/01/18

Image size needs to be divisible by 2:
To allow seamless tiling of the output segmentation map, input tile size needs to be selected such that all 2x2
max pooling operations are applied to a layer with an even x and y size

"""


class MitoSegNet(object):

    def __init__(self, img_rows=656, img_cols=656, org_img_rows=1030, org_img_cols=1300):

        self.img_rows = img_rows
        self.img_cols = img_cols

        self.org_img_rows = org_img_rows
        self.org_img_cols = org_img_cols

    def load_data(self, wmap):

        mydata = Create_npy_files(self.img_rows, self.img_cols)

        if wmap == False:

            imgs_train, imgs_mask_train = mydata.load_train_data(wmap=wmap)
            print(imgs_mask_train.shape)
            return imgs_train, imgs_mask_train

        else:

            imgs_train, imgs_mask_train, imgs_weights = mydata.load_train_data(wmap=wmap)
            print(imgs_mask_train.shape)
            return imgs_train, imgs_mask_train, imgs_weights

    def get_mitosegnet(self, wmap):

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

        # batchnorm architecture
        ######################################################################
        # """

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
        # merge6 = merge([act4, up6], mode='concat', concat_axis=3)

        conv6 = Conv2D(512, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 512))))(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 512))))(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same',
                     kernel_initializer=gauss(stddev=sqrt(2 / (9 * 512))))(UpSampling2D(size=(2, 2))(conv6))

        merge7 = concatenate([conv3, up7], axis=3)
        # merge7 = merge([act3, up7], mode='concat', concat_axis=3)

        conv7 = Conv2D(256, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 256))))(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 256))))(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same',
                     kernel_initializer=gauss(stddev=sqrt(2 / (9 * 256))))(UpSampling2D(size=(2, 2))(conv7))

        merge8 = concatenate([conv2, up8], axis=3)
        # merge8 = merge([act2, up8], mode='concat', concat_axis=3)

        conv8 = Conv2D(128, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 128))))(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 128))))(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same',
                     kernel_initializer=gauss(stddev=sqrt(2 / (9 * 128))))(UpSampling2D(size=(2, 2))(conv8))

        merge9 = concatenate([conv1, up9], axis=3)
        # merge9 = merge([act1, up9], mode='concat', concat_axis=3)

        conv9 = Conv2D(64, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 64))))(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 64))))(conv9)

        conv9 = Conv2D(2, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 64))))(conv9)

        # """
        ######################################################################

        # dropout architecture
        ######################################################################
        """
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=gauss())(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 64))))(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 64))))(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 128))))(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 128))))(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 256))))(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 256))))(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 512))))(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 512))))(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 1024))))(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 1024))))(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 512))))(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 512))))(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 512))))(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 256))))(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 256))))(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 256))))(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 128))))(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 128))))(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 128))))(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 64))))(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 64))))(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 64))))(conv9)
        """
        ######################################################################

        conv10 = Conv2D(1, 1, activation='sigmoid', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 2))))(conv9)

        if wmap == False:
            input = inputs
            loss = self.pixelwise_crossentropy()
        else:
            weights = Input(shape=(self.img_rows, self.img_cols, 1))
            input = [inputs, weights]

            loss = self.weighted_pixelwise_crossentropy(input[1])

        model = Model(input=input, output=conv10)

        model.compile(optimizer=Adam(lr=1e-4), loss=loss, metrics=['accuracy', self.dice_coefficient])

        # model.compile(optimizer=Adam(lr=1e-4), loss=self.pixelwise_crossentropy(input[1]),
        #              metrics=['accuracy', self.dice_coefficient])

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

    def train(self, epochs, wmap):

        print("Loading data")

        if wmap == False:
            imgs_train, imgs_mask_train = self.load_data(wmap=wmap)
        else:
            imgs_train, imgs_mask_train, img_weights = self.load_data(wmap=wmap)

        print("Loading data done")

        model = self.get_mitosegnet(wmap=wmap)
        print("Got mitosegnet")

        if os.path.isfile('data/mitosegnet.hdf5'):

            model.load_weights('data/mitosegnet.hdf5')
            print("Loading weights")

        else:
            print("No previously optimized weights were loaded. Proceeding without")

        # Set network weights saving mode.
        # save previously established network weights (saving model after every epoch)

        print('Fitting model...')

        csv_logger = CSVLogger('data/training_log.csv')

        # Set callback functions to early stop training and save the best model so far
        callbacks = [EarlyStopping(monitor='val_loss', patience=5),
                     ModelCheckpoint(filepath='data/mitosegnet.hdf5', monitor='val_loss', verbose=1,
                                     save_best_only=True),
                     csv_logger]

        if wmap == True:
            x = [imgs_train, img_weights]
            bs = 1

        else:
            x = imgs_train
            bs = 3

        model.fit(x=x, y=imgs_mask_train, batch_size=bs, epochs=epochs, verbose=1,
                  validation_split=0.2, shuffle=True, callbacks=callbacks)

    def predict(self, wmap):

        """
        :return:
        """

        org_img_rows = self.org_img_rows
        org_img_cols = self.org_img_cols
        size = self.img_rows

        mydata = Create_npy_files(self.img_rows, self.img_cols)

        l_imgs = mydata.create_test_data()
        imgs_test = mydata.load_test_data()

        # predict if no npy array exists yet
        if not os.path.isfile("data/results/imgs_mask_test.npy"):
            model = self.get_mitosegnet(wmap=wmap)
            model.load_weights("data/mitosegnet.hdf5")

            print('Predict test data')

            # batch size of 14 was too large to fit into GPU memory
            # verbose: show ongoing progress (0 do not show)
            imgs_mask_test = model.predict(imgs_test, batch_size=4, verbose=1)

            np.save('data/results/imgs_mask_test.npy', imgs_mask_test)

        org_img_list = list(set(list([n.split("_")[1] for n in l_imgs if ".tif" in n])))

        def atoi(text):
            return int(text) if text.isdigit() else text

        def natural_keys(text):
            return [atoi(c) for c in re.split('(\d+)', text)]

        # alphanumeric sorting of list elements
        org_img_list.sort(key=natural_keys)

        # saving arrays as images
        print("Array to image")

        imgs = np.load('data/results/imgs_mask_test.npy')

        img_nr = 0
        org_img_list_index = 0
        for n, image in zip(range(imgs.shape[0]), l_imgs):

            if img_nr == 0:
                current_img = np.zeros((org_img_rows, org_img_cols))

            img = imgs[n]

            # setting any prediction with a probability of 0.5 or lower to 0 and above 0.5 to 1
            img[img > 0.5] = 1
            img[img <= 0.5] = 0

            img = array_to_img(img)

            imgname = image[image.rindex("/") + 1:]
            # print(imgname)

            # stitching the four sub-images back together
            if "0_" in imgname:
                current_img[0:size, 0:size] = img

            elif "1_" in imgname:
                current_img[0:size, org_img_cols - size - 1:org_img_cols - 1] = img

            elif "2_" in imgname:
                current_img[org_img_rows - size - 1:org_img_rows - 1, 0:size] = img

            else:
                current_img[org_img_rows - size - 1:org_img_rows - 1,
                org_img_cols - size - 1:org_img_cols - 1] = img

            img_nr += 1

            # once one image has been fully stitched, remove any objects below 10 px size and save
            if img_nr == 4:
                label_image, num_features = label(current_img)
                new_image = remove_small_objects(label_image, 10)
                new_image[new_image != 0] = 255

                print(org_img_list[org_img_list_index])

                cv2.imwrite("data/results/" + org_img_list[org_img_list_index], new_image)

                org_img_list_index += 1
                img_nr = 0


if __name__ == '__main__':
    mitosegnet = MitoSegNet()

    mitosegnet.train(epochs=20, wmap=True)
    # mitosegnet.predict(wmap=False)

    # K.clear_session()










