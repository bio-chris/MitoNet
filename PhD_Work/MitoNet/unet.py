"""

Partially based on
https://github.com/zhixuhao/unet

>>INSERT DESCRIPTION HERE<<





Written for usage on pcd server (linux)

Christian Fischer

"""

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Activation, BatchNormalization
from keras.optimizers import *
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.initializers import RandomNormal as gauss
from keras.preprocessing.image import array_to_img
from keras import losses
from data import *
import cv2
import os
import shutil
from math import sqrt
from skimage.morphology import remove_small_objects
from scipy.ndimage import label

predict = False


class myUnet(object):

    def __init__(self, img_rows=656, img_cols=656, org_img_rows=1030, org_img_cols=1300):

        self.img_rows = img_rows
        self.img_cols = img_cols

        self.org_img_rows = org_img_rows
        self.org_img_cols = org_img_cols

    def load_data(self):

        # load image data stored in npy format (using function from data.py script)

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

        conv1 = Conv2D(64, 3, padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 64))))(act1)
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

        conv2 = Conv2D(128, 3, padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 128))))(act2)
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

        conv3 = Conv2D(256, 3, padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 256))))(act3)
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

        conv4 = Conv2D(512, 3, padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 512))))(act4)
        batch4 = BatchNormalization()(conv4)
        act4 = Activation("relu")(batch4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(act4)
        ########

        ########
        conv5 = Conv2D(1024, 3, padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 512))))(pool4)
        batch5 = BatchNormalization()(conv5)
        act5 = Activation("relu")(batch5)

        conv5 = Conv2D(1024, 3, padding='same', kernel_initializer=gauss(stddev=sqrt(2 / (9 * 1024))))(act5)
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

        ######################################
        ######################################

        if predict == False:

            print("Training session")
            model = Model(input=inputs, output=conv10)


        else:

            print("Predict session")
            model = Model(input=inputs, output=conv10)

        model.compile(optimizer=Adam(lr=1e-4), loss=self.weighted_pixelwise_crossentropy(inputs),
                      metrics=['accuracy', self.dice_coefficient])

        return model

    def dice_coefficient(self, y_true, y_pred):

        # calculate dice coefficient based on y_true, y_pred tensors

        smooth = 1

        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)

        intersection = K.sum(y_true_f * y_pred_f)

        dice = -(2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

        return dice

    def weighted_pixelwise_crossentropy(self):

        def loss(y_true, y_pred):
            return losses.binary_crossentropy(y_true, y_pred)

        return loss

    def train(self):

        global predict
        global run

        predict = False

        print("loading data")
        imgs_train, imgs_mask_train = self.load_data()

        print("loading data done")

        model = self.get_unet()
        print("got unet")

        if run == 1:
            print("No previously optimized weights were loaded. Proceeding without")

        elif run == 2:

            model.load_weights('data/unet_1.hdf5')
            print("Loading weights")

        else:

            model.load_weights('data/unet_2.hdf5')
            print("Loading weights")

        print('Fitting model...')

        def run_training(run_nr):

            if run_nr == 1:
                epochs = 1

            elif run_nr == 2:
                epochs = 9

            else:
                epochs = 10

            model_checkpoint = ModelCheckpoint("data/unet_" + run_nr + ".hdf5", monitor='loss', verbose=1,
                                               save_best_only=True)

            history = model.fit(x=imgs_train, y=imgs_mask_train, batch_size=3, epochs=epochs, verbose=1,
                                validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])

            pd.DataFrame(history.history).to_csv("data/history_" + run_nr + ".csv")

        run_training(run)

    def predict(self):

        """
        :return:
        """

        global predict
        global run

        predict = True

        mydata = dataProcess(self.img_rows, self.img_cols)

        l_imgs = mydata.create_test_data()
        imgs_test = mydata.load_test_data()

        model = self.get_unet()

        model.load_weights("data/unet_" + run + ".hdf5")

        print('predict test data')
        imgs_mask_test = model.predict(imgs_test, batch_size=12, verbose=1)

        np.save('data/results/imgs_mask_test.npy', imgs_mask_test)

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

        path = "data/results"
        merge_path = "data/results/merged"

        if os.path.lexists(merge_path):
            pass

        else:
            os.mkdir(merge_path)

        img_list = os.listdir(path)
        # list comprehension: removing sub-image values and storing  original image names in list

        org_img_list = set(list([n.split("_")[1] for n in img_list if ".tif" in n]))

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

        def merge():

            global i

            for folder in org_img_list:

                if ".tif" in folder:

                    folder_img = os.listdir(path + "/" + folder)
                    current_img = np.zeros((1030, 1300))

                    for img in folder_img:

                        # img_name = img.split("_")[1] + "_" + img.split("_")[2]
                        img_name = img.split("_")[1]

                        pic = cv2.imread(path + "/" + folder + "/" + img, cv2.IMREAD_GRAYSCALE)

                        if "0_" in img:
                            # y, x
                            current_img[0:656, 0:656] = pic

                        elif "1_" in img:
                            current_img[0:656, 643:1299] = pic

                        elif "2_" in img:
                            current_img[373:1029, 0:656] = pic

                        else:
                            current_img[373:1029, 643:1299] = pic

                    ### new section (01/10/18): removal of particles smaller than 10 px area
                    #####################

                    label_image, num_features = label(current_img)

                    # print(num_features)

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

        def predict_test_accuracy():

            pred_label = "data/results"
            test_label = "data/test_label"

            print("\n")

            for image in os.listdir(pred_label):

                if ".tif" in image:
                    true = cv2.imread(test_label + "/" + image, cv2.IMREAD_GRAYSCALE)
                    pred = cv2.imread(pred_label + "/" + image, cv2.IMREAD_GRAYSCALE)

                    # sorensen dice similarity coefficient
                    dice = (np.sum(pred[true == 255]) * 2.0) / (np.sum(pred) + np.sum(true))

                    print(image, dice)

        def move_files():

            global run

            path = "data/results"
            dest_path = "data/results_storage"

            if run == 1:
                os.makedirs("data/results_storage/1_epoch")
            elif run == 2:
                os.makedirs("data/results_storage/10_epochs")
            elif run == 3:
                os.makedirs("data/results_storage/20_epochs")

            for img in os.listdir(path):

                if run == 1:
                    # source, destination
                    shutil.move(path + "/" + img, dest_path + "/1_epoch/" + img)

                elif run == 2:
                    # source, destination
                    shutil.move(path + "/" + img, dest_path + "/10_epochs/" + img)

                elif run == 3:
                    # source, destination
                    shutil.move(path + "/" + img, dest_path + "/20_epochs/" + img)

        group()
        merge()
        del_move()
        move_files()


if __name__ == '__main__':
    global run

    myunet = myUnet()

    for r in [1, 2, 3]:
        run = r

        myunet.train()
        myunet.predict()

        K.clear_session()



