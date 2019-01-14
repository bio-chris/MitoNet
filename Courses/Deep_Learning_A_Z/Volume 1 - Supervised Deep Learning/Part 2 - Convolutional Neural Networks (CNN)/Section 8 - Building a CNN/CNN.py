"""

Convolution > Max pooling > Flattening > Full connection

Convolution:
__________________

integration of two functions

multiplying the input image with a feature detector (or kernel (usually 3x3, but not always)) = feature map

convolutional layer: multiple feature maps (due to usage of different filters on input image)


ReLU Layer:
__________________

rectifier function is applied to convolutional layers


Max pooling:
__________________

feature map (5x5) (using a 2x2 box) > pooled feature map  (3x3)


Flattening:
__________________

flattening turns a 2d array into a 1d array

necessary, to feed into artificial neural network (creating the input layers)


-------

In a CNN, during backpropagation not only the weights are adjusted but also the feature detectors (kernel)


Summary:

Input image > convolution > relu > pooling > flattening > fully connected artificial neural network

training through forward and backpropagation process (weights and feature maps are adjusted during training)


Softmax and Cross-Entropy


Binary classification output can be any (arbitrary) two values

Apply softmax function which would make both values add up to 1 (or sigmoid function for example)

Percentage values given to cross entropy function will output 0 or 1

(Example: 90% for A, 10% for B, so A = 1, B = 0


"""

"""

Aim of CNN will be to recognize cats and dogs in images 

"""

from keras.models import Sequential, Input
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import os


# initialising the CNN

def CNN():

    classifier = Sequential()

    # step 1: Convolution

    # create 32 feature detectors, using a 3x3 kernel
    # tensorflow backend its 64, 64, 3 (width, height, channels) / for theano backend it is the opposite
    classifier.add(Conv2D(32, 3,3, input_shape=(64, 64, 3), activation= "relu"))

    # step 2: Max pooling

    classifier.add(MaxPooling2D(pool_size= (2,2)))

    # second Convolution and Max pooling step
    classifier.add(Conv2D(32, 3,3, activation= "relu"))
    classifier.add(MaxPooling2D(pool_size= (2,2)))


    # step 3: Flattening

    classifier.add(Flatten())

    # step 4: fully connected layer

    # adding input and first hidden layer with dropout (randomly deactivates neurons to prevent overfitting)
    classifier.add(Dense(units=128, activation='relu'))

    # adding the output layer
    classifier.add(Dense(units=1, activation='sigmoid'))

    classifier.compile(optimizer= "adam", loss= "binary_crossentropy", metrics=['accuracy'])

    return classifier



# fitting the cnn to the image 

def train():

    train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    training_set = train_datagen.flow_from_directory("dataset/training_set",
                                                     target_size=(64, 64),
                                                     batch_size=32,
                                                     class_mode="binary")

    test_set = test_datagen.flow_from_directory("dataset/test_set",
                                                target_size=(64,64),
                                                batch_size=32,
                                                class_mode="binary")

    classifier = CNN()

    if os.path.isfile('dataset/weights.h5'):

        classifier.load_weights('dataset/weights.h5')
        print("Loading weights")

    else:

        print("No previously optimized weights were loaded. Proceeding without")

    # Set network weights saving mode.
    # save previously established network weights (saving model after every epoch)
    model_checkpoint = ModelCheckpoint('dataset/weights.h5', monitor='loss', verbose=1, save_best_only=True)


    classifier.fit_generator(training_set,
                        samples_per_epoch=8000,
                        validation_data=test_set,
                        nb_val_samples=2000,
                        callbacks=[model_checkpoint])


    #classifier.save_weights("dataset/weights.h5")



def predict():

    img1 = image.load_img("dataset/single_prediction/cat_or_dog_1.jpg", target_size=(64,64))

    img1 = image.img_to_array(img1)
    img1 = np.expand_dims(img1, axis= 0)

    classifier = CNN()

    classifier.load_weights('dataset/weights.h5')

    # class indices: cats = 0, dog = 1

    result = classifier.predict(img1)

    print(result)

#train()
predict()

