#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Referece: DeepUNet: A Deep Fully Convolutional Network for Pixel-level Sea-Land Segmentation, Li et al., 2017
from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, ZeroPadding2D, Conv2DTranspose, UpSampling2D
from keras.layers.merge import concatenate, add, average
from keras.layers import LeakyReLU, BatchNormalization, Activation, Dropout, MaxPooling2D
from keras import regularizers

class UNet(object):
    def __init__(self, input_channel_count, output_channel_count, first_layer_filter_count):
        self.INPUT_IMAGE_SIZE = 256
        self.CONCATENATE_AXIS = -1
        #self.CONV_FILTER_SIZE = 4
        #self.CONV_STRIDE = 2
        #self.CONV_PADDING = (1, 1)
        #self.DECONV_FILTER_SIZE = 2
        #self.DECONV_STRIDE = 2

        # (256 x 256 x input_channel_count)
        inputs = Input((self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE, input_channel_count))

        # Encoder part: エンコーダーの作成
        # First layer: 1層目は構造が違っていたのでベタ書き: 256 => 128
        enc1 = inputs
        # Residual part
        res_enc1 = enc1
        # BN => Ac => Conv
        res_enc1 = BatchNormalization()(res_enc1)
        res_enc1 = Activation(activation='relu')(res_enc1)
        res_enc1 = Conv2D(int(first_layer_filter_count*2), 3, strides=1,padding="same",kernel_initializer='he_uniform')(res_enc1)

        res_enc1 = BatchNormalization()(res_enc1)
        res_enc1 = Activation(activation='relu')(res_enc1)
        res_enc1 = Conv2D(first_layer_filter_count, 3, strides=1,padding="same",kernel_initializer='he_uniform')(res_enc1)

        # Shortcut part
        shortcut_enc1 = enc1
        shortcut_enc1 = Conv2D(first_layer_filter_count, 1, strides=1,padding="same")(shortcut_enc1)

        # Adding
        add_enc1 = add([res_enc1,shortcut_enc1])
        add_enc1 = Activation(activation='relu')(add_enc1)

        # Using stride to reduce image size instead of using maxpooling
        enc1 = Conv2D(first_layer_filter_count, 2, strides=2,padding="same")(add_enc1) # 256 => 128

        # follwing encoder layers: 2層目以降
        filter_count = first_layer_filter_count # 32
 
        # follwing encoder layers: 2層目以降
        enc2, res_enc2 = self._add_encoding_layer(filter_count,    enc1, True) # 128 => 64
        enc3, res_enc3 = self._add_encoding_layer(filter_count*1,  enc2, True) # 64 => 32
        enc4, res_enc4 = self._add_encoding_layer(filter_count*2,  enc3, True) # 32 => 16
        enc5, res_enc5 = self._add_encoding_layer(filter_count*4,  enc4, True) # 16 => 8
        enc6, res_enc6 = self._add_encoding_layer(filter_count*8,  enc5, True) # 8 => 4
        enc7, res_enc7 = self._add_encoding_layer(filter_count*16, enc6, False) # 4 => 4

        # Decoder part:
        dec2 = self._add_decoding_layer(filter_count*16, False, enc7, res_enc7, True) # 4 => 8
        dec3 = self._add_decoding_layer(filter_count*8,  False, dec2, res_enc6, True) # 8 => 16
        dec4 = self._add_decoding_layer(filter_count*4,  True,  dec3, res_enc5, True) # 16 => 32
        dec5 = self._add_decoding_layer(filter_count*2,  True,  dec4, res_enc4, True) # 32 => 64
        dec6 = self._add_decoding_layer(filter_count*1,  True,  dec5, res_enc3, True) # 64 => 128
        dec7 = self._add_decoding_layer(filter_count,    True,  dec6, res_enc2, True) # 128 => 256

        # Output layer with softmax or sigmoid activation : This layer is simpler than original in the reference
        dec8 = concatenate([dec7, res_enc1], axis=self.CONCATENATE_AXIS)
        dec8 = Conv2D(filter_count, 3, strides=1,padding="same",kernel_initializer='he_uniform')(dec8)
        dec8 = BatchNormalization()(dec8)
        dec8 = Activation(activation='relu')(dec8)

        dec8 = Conv2D(output_channel_count,1,strides=1,padding="same")(dec8)
        dec8 = Activation(activation='sigmoid')(dec8)

        self.UNET = Model(inputs=inputs, outputs=dec8)


    def _add_encoding_layer(self, filter_count, sequence, mp):

        # Residual part
        res_sequence = sequence

        res_sequence = BatchNormalization()(res_sequence)
        res_sequence = Activation(activation='relu')(res_sequence)
        res_sequence = Conv2D(filter_count, 3, strides=1,padding="same",kernel_initializer='he_uniform')(res_sequence)

        res_sequence = BatchNormalization()(res_sequence)
        res_sequence = Activation(activation='relu')(res_sequence)
        res_sequence = Conv2D(filter_count, 3, strides=1,padding="same",kernel_initializer='he_uniform')(res_sequence)

        # shortcut part
        shortcut_sequence = sequence
        # 1x1 projection
        shortcut_sequence = Conv2D(filter_count, 1, strides=1,padding="same")(shortcut_sequence)

        # add & export
        add_sequence = add([res_sequence, shortcut_sequence])
        add_sequence = Activation(activation='relu')(add_sequence)

        if mp:
            # Reducing size with stride
            new_sequence = Conv2D(filter_count, 2, strides=2,padding="same")(add_sequence)
        else:
            new_sequence = Conv2D(filter_count, 1, strides=1,padding="same")(add_sequence)
            
        return new_sequence, add_sequence

    
    def _add_decoding_layer(self, filter_count, add_drop_layer, sequence, res_enc, us):

        # Residual part
        res_sequence = sequence
        # import & concatenate
        res_sequence = concatenate([res_sequence, res_enc], axis=self.CONCATENATE_AXIS)

        res_sequence = BatchNormalization()(res_sequence)
        res_sequence = Activation(activation='relu')(res_sequence)
        res_sequence = Conv2D(int(filter_count*2), 3, strides=1,padding="same",kernel_initializer='he_uniform')(res_sequence)
        
        # In original papre, kernel size set to be  2, but in the author's github, the kernel size = 3.
        res_sequence = BatchNormalization()(res_sequence)
        res_sequence = Activation(activation='relu')(res_sequence)
        res_sequence = Conv2D(filter_count, 3, strides=1,padding="same",kernel_initializer='he_uniform')(res_sequence)
        
        # shortcut part
        shortcut_sequence = sequence
        # 1x1 projection
        shortcut_sequence = Conv2D(filter_count, 1, strides=1,padding="same")(shortcut_sequence)

        # add
        add_sequence = add([res_sequence, shortcut_sequence])
        add_sequence = Activation(activation='relu')(add_sequence)

        # Dropout?
        if add_drop_layer:
            add_sequence = Dropout(0.2)(add_sequence)


        if us:
            # Replacing Upsampling with deconvolution
            new_sequence = Conv2DTranspose(filter_count, 2, strides=2,padding="same",kernel_initializer='he_uniform')(add_sequence)
        else:
            new_sequence = Conv2D(filter_count, 1, strides=1,padding="same")(add_sequence)
            
        return new_sequence


    def get_model(self):
        return self.UNET
