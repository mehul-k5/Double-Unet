import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.applications import *
from tensorflow.keras.models import Model

def squeezeExciteBlock(inputs, ratio=8):
    init = inputs
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)
    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    x = Multiply()([init, se])
    return x

def convolutionBlock(inputs, filters):
    x = inputs
    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = squeezeExciteBlock(x)
    return x

def encoder1(inputs):
    skip_connections = []
    model = VGG19(include_top=False, weights='imagenet', input_tensor=inputs)
    names = ["block1_conv2", "block2_conv2", "block3_conv4", "block4_conv4"]
    for name in names:
        skip_connections.append(model.get_layer(name).output)
    output = model.get_layer("block5_conv4").output
    return output, skip_connections

def decoder1(inputs, skip_connections):
    num_filters = [256, 128, 64, 32]
    skip_connections.reverse()
    x = inputs
    for i, f in enumerate(num_filters):
        x = UpSampling2D((2, 2), interpolation='bilinear')(x)
        x = Concatenate()([x, skip_connections[i]])
        x = convolutionBlock(x, f)
    return x

def encoder2(inputs):
    num_filters = [32, 64, 128, 256]
    skip_connections = []
    x = inputs
    for i, f in enumerate(num_filters):
        x = convolutionBlock(x, f)
        skip_connections.append(x)
        x = MaxPool2D((2, 2))(x)
    return x, skip_connections

def decoder2(inputs, skip_1, skip_2):
    num_filters = [256, 128, 64, 32]
    skip_2.reverse()
    x = inputs
    for i, f in enumerate(num_filters):
        x = UpSampling2D((2, 2), interpolation='bilinear')(x)
        x = Concatenate()([x, skip_1[i], skip_2[i]])
        x = convolutionBlock(x, f)
    return x


def outputBlock(inputs):
    x = Conv2D(1, (1, 1), padding="same")(inputs)
    x = Activation('sigmoid')(x)
    return x

def Upsample(tensor, size):
    #Bilinear upsampling#
    def _upsample(x, size):
        return tf.image.resize(images=x, size=size)
    return Lambda(lambda x: _upsample(x, size), output_shape=size)(tensor)


def ASPP(x, filter):
    shape = x.shape
    p1 = AveragePooling2D(pool_size=(shape[1], shape[2]))(x)
    p1 = Conv2D(filter, 1, padding="same")(p1)
    p1 = BatchNormalization()(p1)
    p1 = Activation("relu")(p1)
    p1 = UpSampling2D((shape[1], shape[2]), interpolation='bilinear')(p1)
    p2 = Conv2D(filter, 1, dilation_rate=1, padding="same", use_bias=False)(x)
    p2 = BatchNormalization()(p2)
    p2 = Activation("relu")(p2)
    p3 = Conv2D(filter, 3, dilation_rate=6, padding="same", use_bias=False)(x)
    p3 = BatchNormalization()(p3)
    p3 = Activation("relu")(p3)
    p4 = Conv2D(filter, 3, dilation_rate=12, padding="same", use_bias=False)(x)
    p4 = BatchNormalization()(p4)
    p4 = Activation("relu")(p4)
    p5 = Conv2D(filter, 3, dilation_rate=18, padding="same", use_bias=False)(x)
    p5 = BatchNormalization()(p5)
    p5 = Activation("relu")(p5)
    p = Concatenate()([p1, p2, p3, p4, p5])
    p = Conv2D(filter, 1, dilation_rate=1, padding="same", use_bias=False)(p)
    p = BatchNormalization()(p)
    p = Activation("relu")(p)
    return p

def build_model(shape):
    """returns a double Unet model of given shape"""
    inputs = Input(shape)
    x, skip_1 = encoder1(inputs)
    x = ASPP(x, 64)
    x = decoder1(x, skip_1)
    outputs1 = outputBlock(x)
    x = inputs * outputs1
    x, skip_2 = encoder2(x)
    x = ASPP(x, 64)
    x = decoder2(x, skip_1, skip_2)
    outputs2 = outputBlock(x)
    outputs = Concatenate()([outputs1, outputs2])
    model = Model(inputs, outputs)
    return model

if __name__ == "__main__":
    model = build_model((288,384,3))
    model.summary()
