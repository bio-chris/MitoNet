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
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Activation, BatchNormalization
from keras.optimizers import *
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.initializers import RandomNormal as gauss
from keras.preprocessing.image import array_to_img
from keras import losses
from Training_DataGenerator import *
import cv2
import os
import shutil
from math import sqrt
from skimage.morphology import remove_small_objects
from scipy.ndimage import label

"""

16/01/18

Image size needs to be divisible by 2:
To allow seamless tiling of the output segmentation map, input tile size needs to be selected such that all 2x2
max pooling operations are applied to a layer with an even x and y size

"""

predict = False


class myUnet(object):

    # def __init__(self, img_rows=656, img_cols=656, org_img_rows=1030, org_img_cols=1300):
    def __init__(self, img_rows=448, img_cols=448, org_img_rows=675, org_img_cols=884):

        self.img_rows = img_rows
        self.img_cols = img_cols

        self.org_img_rows = org_img_rows
        self.org_img_cols = org_img_cols

    def load_data(self):

        mydata = dataProcess(self.img_rows, self.img_cols)

        imgs_train, imgs_mask_train = mydata.load_train_data()
        print(imgs_mask_train.shape)

        return imgs_train, imgs_mask_train

    def get_unet(self):

        global predict

        inputs = Input(shape=(self.img_rows, self.img_cols, 1))
        print(inputs.get_shape(), type(inputs))

        # core u-net architecture
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
        merge6 = merge([conv4, up6], mode='concat', concat_axis=3)

        conv6 = Conv2D(512, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 512))))(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 512))))(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same',
                     kernel_initializer=gauss(stddev=sqrt(2 / (9 * 512))))(UpSampling2D(size=(2, 2))(conv6))
        merge7 = merge([conv3, up7], mode='concat', concat_axis=3)

        conv7 = Conv2D(256, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 256))))(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 256))))(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same',
                     kernel_initializer=gauss(stddev=sqrt(2 / (9 * 256))))(UpSampling2D(size=(2, 2))(conv7))

        merge8 = merge([conv2, up8], mode='concat', concat_axis=3)

        conv8 = Conv2D(128, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 128))))(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 128))))(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same',
                     kernel_initializer=gauss(stddev=sqrt(2 / (9 * 128))))(UpSampling2D(size=(2, 2))(conv8))
        merge9 = merge([conv1, up9], mode='concat', concat_axis=3)

        conv9 = Conv2D(64, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 64))))(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 64))))(conv9)

        conv9 = Conv2D(2, 3, activation='relu', padding='same',
                       kernel_initializer=gauss(stddev=sqrt(2 / (9 * 64))))(conv9)

        conv10 = Conv2D(1, 1, activation='sigmoid', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 2))))(conv9)

        """
        if predict == False:

            print("Training session")

            model = Model(input=inputs, output=conv10)


        else:

            print("Predict session")

            model = Model(input=inputs, output=conv10)
        """

        model = Model(input=inputs, output=conv10)

        # lr: learning rate
        model.compile(optimizer=Adam(lr=1e-4), loss=self.pixelwise_crossentropy(),
                      metrics=['accuracy', self.dice_coefficient])

        return model

    def dice_coefficient(self, y_true, y_pred):

        smooth = 1

        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)

        intersection = K.sum(y_true_f * y_pred_f)

        # dice = -(2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

        return dice

    def pixelwise_crossentropy(self):

        def loss(y_true, y_pred):
            return losses.binary_crossentropy(y_true, y_pred)

        return loss

    def train(self, epochs):

        global predict

        predict = False

        print("loading data")

        imgs_train, imgs_mask_train = self.load_data()
        print("loading data done")

        model = self.get_unet()
        print("got unet")

        if os.path.isfile('data/unet.hdf5'):

            model.load_weights('data/unet.hdf5')
            print("Loading weights")

        else:
            print("No previously optimized weights were loaded. Proceeding without")
            # exit()

        # Set network weights saving mode.
        # save previously established network weights (saving model after every epoch)

        print('Fitting model...')

        model_checkpoint = ModelCheckpoint("data/unet.hdf5", monitor='loss', verbose=1, save_best_only=True)

        history = model.fit(x=imgs_train, y=imgs_mask_train, batch_size=3, epochs=epochs, verbose=1,
                            validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])

        """

        # Set callback functions to early stop training and save the best model so far
        callbacks = [EarlyStopping(monitor='val_loss', patience=2),
                    ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

        # Train neural network
        history = network.fit(train_features, # Features
                              train_target, # Target vector
                              epochs=20, # Number of epochs
                              callbacks=callbacks, # Early stopping
                              verbose=0, # Print description after each epoch
                              batch_size=100, # Number of observations per batch
                              validation_data=(test_features, test_target)) # Data for evaluation

        """

        pd.DataFrame(history.history).to_csv("data/history.csv")

    def predict(self):

        """
        :return:
        """

        org_img_rows = self.org_img_rows
        org_img_cols = self.org_img_cols

        size = self.img_rows

        mydata = dataProcess(self.img_rows, self.img_cols)

        l_imgs = mydata.create_test_data()
        imgs_test = mydata.load_test_data()

        model = self.get_unet()
        model.load_weights("data/MitoNet_Final.hdf5")

        print('predict test data')

        # batch size of 14 was too large to fit into GPU memory
        # verbose: show ongoing progress (0 do not show)
        imgs_mask_test = model.predict(imgs_test, batch_size=4, verbose=1)

        np.save('data/results/imgs_mask_test.npy', imgs_mask_test)

        # saving arrays as images
        print("array to image")

        imgs = np.load('data/results/imgs_mask_test.npy')

        for n, image in zip(range(imgs.shape[0]), l_imgs):
            img = imgs[n]

            img[img > 0.5] = 1
            img[img <= 0.5] = 0

            img = array_to_img(img)

            imgname = image[image.rindex("/") + 1:]

            img.save("data/results/" + imgname)

        os.remove('data/results/imgs_mask_test.npy')

        # added on 04/04/18 to stitch images back together
        ##############################

        path = "data/results"

        merge_path = "data/results/merged"

        if os.path.lexists(merge_path):
            pass

        else:
            os.mkdir(merge_path)

        img_list = os.listdir(path)
        # list comprehension: removing sub-image values and storing  original image names in list

        org_img_list = set(list([n.split("_")[1] for n in img_list if ".tif" in n]))
        # org_img_list = img_list

        """
        generates list containing names of original (fully sized) images
        creates each one folder named after original image in which all sub-images will be moved
        """

        def group():

            for img in org_img_list:

                if os.path.lexists(path + "/" + img):
                    print(img + ' already exists')

                else:
                    os.mkdir(path + "/" + img)

            for img in img_list:

                if ".tif" in img:
                    img_name = img.split("_")[1]

                    # source, destination
                    shutil.move(path + "/" + img, path + "/" + img_name + "/" + img)

        """
        iterate through each sub-image in every folder
        create empty 2-d array and fill it with sections until complete
        save new image in merged folder
        """

        def merge(org_img_rows, org_img_cols):

            global i

            for folder in org_img_list:

                if ".tif" in folder:

                    folder_img = os.listdir(path + "/" + folder)
                    current_img = np.zeros((org_img_rows, org_img_cols))

                    for img in folder_img:

                        img_name = img.split("_")[1]

                        pic = cv2.imread(path + "/" + folder + "/" + img, cv2.IMREAD_GRAYSCALE)

                        if "0_" in img:
                            current_img[0:size, 0:size] = pic

                        elif "1_" in img:
                            current_img[0:size, org_img_cols - size - 1:org_img_cols - 1] = pic

                        elif "2_" in img:
                            current_img[org_img_rows - size - 1:org_img_rows - 1, 0:size] = pic

                        else:
                            current_img[org_img_rows - size - 1:org_img_rows - 1,
                            org_img_cols - size - 1:org_img_cols - 1] = pic

                    ### new section (01/10/18): removal of particles smaller than 10 px area
                    #####################

                    label_image, num_features = label(current_img)

                    new_image = remove_small_objects(label_image, 10)

                    new_image[new_image != 0] = 255

                    #####################

                    cv2.imwrite(merge_path + "/" + img_name, new_image)

        def del_move():

            # delete folders
            for folder in org_img_list:

                if ".tif" in folder:
                    shutil.rmtree(path + "/" + folder)

            # move files from merged to upper directory level
            for img in os.listdir(merge_path):
                # source, destination
                shutil.move(merge_path + "/" + img, path + "/" + img)

            os.rmdir(merge_path)

        ##############################

        group()
        merge(org_img_rows, org_img_cols)
        del_move()


if __name__ == '__main__':
    myunet = myUnet()

    #myunet.train(epochs=1)
    myunet.predict()

    # K.clear_session()










