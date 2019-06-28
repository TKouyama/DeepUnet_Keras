##
## Import Modules ##
##
import os
import numpy as np

## Keras
import keras.backend as K
from keras.optimizers import Adam
from keras import losses
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

## Unet model (assuming unte.py is in the same directory)
#from unet import UNet
#from unet_deep_simple import UNet
from unet_deep_modified import UNet

IMAGE_SIZE = 256

# Normalize pixel-value from -1 to 1 for original images
# assuming image has 0-255 values
def normalize_x(image):
    image = image/127.5 -1
    return image

# Convert normalized value to 0 to 255
def denormalize_x(image):
    image = image*127.5 + 127.5
    return image


# Normalize pixel-value from 0 to 1 for label images
# assuming label image has 0-255
def normalize_y(image):
    image = image/255
    return image

# Convert normalized value to 0 to 255
def denormalize_y(image):
    image = image*255
    return image

# Load input images
def load_X(folder_path, input_channel):
    import os, cv2

    image_files = os.listdir(folder_path)
    image_files.sort()

    image = np.zeros((IMAGE_SIZE,IMAGE_SIZE,3),np.float32)
    images = np.zeros((len(image_files), IMAGE_SIZE, IMAGE_SIZE, input_channel), np.float32)

    for i, image_file in enumerate(image_files):
        image = cv2.imread(folder_path + os.sep + image_file)

        # Test: Read only specied number of channels. / up to 3 channels
        image = image[:,:,:input_channel]

        # Resize image patches with keeping same aspect ratio
        im_x = int(image.shape[1])
        im_y = int(image.shape[0])
        if im_y > IMAGE_SIZE or im_x > IMAGE_SIZE:
            im_x =int(im_x/2)
            im_y =int(im_y/2)

            # In case that im_x or im_y is still lager than IMAGE_SIZE
            if im_y > IMAGE_SIZE or im_x > IMAGE_SIZE:
                im_x =int(im_x/2)
                im_y =int(im_y/2)

            # Resizing
            image = cv2.resize(image, (im_x, im_y), interpolation = cv2.INTER_AREA)

        # Centering + Normalizing images
        st_x = int(IMAGE_SIZE/2 - im_x/2)
        st_y = int(IMAGE_SIZE/2 - im_y/2)
        en_x = st_x + im_x
        en_y = st_y + im_y
        images[i,st_y:en_y,st_x:en_x,:] = normalize_x(image)

    return images, image_files


# Load label images
def load_Y(folder_path):
    import os, cv2

    image_files = os.listdir(folder_path)
    image_files.sort()
    images = np.zeros((len(image_files), IMAGE_SIZE, IMAGE_SIZE, 1), np.float32)
    for i, image_file in enumerate(image_files):
        image = cv2.imread(folder_path + os.sep + image_file, cv2.IMREAD_GRAYSCALE)
        im_x = int(image.shape[1])
        im_y = int(image.shape[0])

        # Resize with keeping same aspect ratio
        if im_y > IMAGE_SIZE or im_x > IMAGE_SIZE:
            im_x =int(im_x/2)
            im_y =int(im_y/2)

            # In case that im_x or im_y is still lager than IMAGE_SIZE
            if im_y > IMAGE_SIZE or im_x > IMAGE_SIZE:
                im_x =int(im_x/2)
                im_y =int(im_y/2)
            image = cv2.resize(image, (im_x, im_y), interpolation = cv2.INTER_AREA)

        image = image[:, :, np.newaxis]

        # Centering + Normalizing images
        st_x = int(IMAGE_SIZE/2 - im_x/2)
        st_y = int(IMAGE_SIZE/2 - im_y/2)
        en_x = st_x + im_x
        en_y = st_y + im_y
        images[i,st_y:en_y,st_x:en_x,:] = normalize_y(image)

    return images


## function for measuring a dice coefficient
def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return 2.0 * intersection / (K.sum(y_true) + K.sum(y_pred) + 1.)

## function for measuring loss value
def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

## Merge loss
def bce_dice_loss(y_true, y_pred):
    a = 0.5
    b = 1-a
    loss = a * binary_crossentropy(y_true, y_pred) + b * dice_loss(y_true, y_pred)
    return loss

#
# function for training U-net
#
def train_unet(input_dir_train, check_dir_train, input_channel_count, Batch_size, First_fileter_layer_count, Num_epoch):
    # Number of output layer is 1 (Fixed)
    output_channel_count = 1

    print("Data load...")
    X_train, file_names = load_X(input_dir_train+os.sep+ 'left_images',input_channel_count)

    # trainingDataフォルダ配下にleft_groundTruthフォルダを置いている
    Y_train = load_Y(input_dir_train + os.sep + 'left_groundTruth')

    print("Model define...")

    # Generate U-Net
    network = UNet(input_channel_count, output_channel_count, First_layer_filter_count)
    model = network.get_model()

    # Use (1-Dice coefficient) as loss, Adam as optimizer
    #model.compile(loss = dice_coef_loss, optimizer=Adam(),metrics=[dice_coef])
    #model.compile(loss = 'binary_crossentropy', optimizer=Adam(),metrics=['acc'])
    model.compile(loss = bce_dice_loss, optimizer=Adam(),metrics=[dice_coef])

    ## for using pre-trained model
    #print("Load weight...")
    #model.load_weights(check_dir_train+'unet_weights.hdf5')

    # Training
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        #vertical_flip=True,
        #zoom_ratio=(1.,1.5),
        validation_split=0.1)
    val_gen = datagen.flow(X_train, Y_train,subset='validation')

    #datagen.fit(X_train)
    history = model.fit_generator(
        datagen.flow(X_train, Y_train, batch_size=Batch_size,subset='training'),
        validation_data=val_gen,
        validation_steps=(len(X_train)*0.1)//Batch_size,
        steps_per_epoch= (len(X_train)*0.9)//Batch_size,
        epochs=Num_epoch,
        verbose=1)

    # Save progress
    print("Save...")
    model.save_weights(check_dir_train+os.sep+'unet_weights.hdf5')

#
# function for adopting trained U-net to an image
#
def predict(input_dir_test, model_dir_test, result_dir_test, input_channel_count, Batch_size, First_layer_filter_count):
    # Import Opencv to save result images
    import cv2


    X_test, file_names = load_X(input_dir_test + os.sep + 'left_images', input_channel_count)    

    output_channel_count = 1
    network = UNet(input_channel_count, output_channel_count, First_layer_filter_count)
    model = network.get_model()
    model.load_weights(model_dir_test + os.sep + 'unet_weights.hdf5')

    Y_pred = model.predict(X_test, Batch_size)
    # Note: Y_pred has an image size of 256x256

    # Output into result_dir_test
    for i, y in enumerate(Y_pred):
        img = cv2.imread(input_dir_test + os.sep + 'left_images' + os.sep + file_names[i])

        if i < 10:
            cv2.imwrite(result_dir_test+os.sep+'prediction00' + str(i) + '_o.png', denormalize_x(X_test[i]))
            cv2.imwrite(result_dir_test+os.sep+'prediction00' + str(i) + '_p.png', denormalize_y(y))
        if i >= 10 and i < 100:
            cv2.imwrite(result_dir_test+os.sep+'prediction0' + str(i) + '_o.png', denormalize_x(X_test[i]))
            cv2.imwrite(result_dir_test+os.sep+'prediction0' + str(i) + '_p.png', denormalize_y(y))
        if i >= 100:
            cv2.imwrite(result_dir_test+os.sep+'prediction' + str(i) + '_o.png', denormalize_x(X_test[i]))
            cv2.imwrite(result_dir_test+os.sep+'prediction' + str(i) + '_p.png', denormalize_y(y))

#
# main
#
if __name__ == '__main__':

    # 必要なだけGPU memoryを確保する
    import tensorflow as tf
    from keras.backend import tensorflow_backend

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    tensorflow_backend.set_session(session)

    #
    # Set shared parameters in training and prediction
    #

    # Number of input channels: 3 channels = RGB
    input_channel_count = 3

    # Batch size 
    Batch_size = 16 # 32

    # for deep unet
    First_layer_filter_count = 32

    #
    # Train #
    #
    print("Training...")
    input_dir_train = './test_datasets/data/trainData/'
    check_dir_train = './check_points/'
    Num_epoch = 150
    #Num_epoch = 60

    train_unet(input_dir_train, check_dir_train, input_channel_count, Batch_size, First_layer_filter_count, Num_epoch)

    #
    # Prediction #
    #
    print("Prediction...")
    input_dir_test = './test_datasets/data/testData/'
    model_dir_test = './check_points/'
    result_dir_test = './results/'
    
    predict(input_dir_test, model_dir_test, result_dir_test, input_channel_count, Batch_size, First_layer_filter_count)

