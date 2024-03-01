"""
Convolutional neural network

"""

from typing import Union

import tensorflow as tf
from keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    concatenate,
    Conv2DTranspose,
    Dropout,
)
from keras.models import Model


def multi_unet_model(n_classes=4, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1):
    # Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    # s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    # Contraction path
    c1 = Conv2D(
        16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(
        16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(
        32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(
        32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(
        128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(
        128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(
        256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(
        256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c5)

    # Expansive path
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(
        128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(
        128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(
        32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(
        32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(
        16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(
        16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c9)

    outputs = Conv2D(n_classes, (1, 1), activation="softmax")(c9)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model


def conv_block(
    input_layer: tf.Tensor,
    *,
    n_filters: int,
    kernel_size: tuple[int, int],
    dropout: float,
) -> tf.Tensor:
    """
    A bulding block for the U-Net model.

    Represents two layers in the network: performs convolution and activation, dropout, then convolution and activation again.
    Uses ReLU activation, he_normal initialistaion and same padding.

    :param input_layer: the activations of the previous layer (or the input if this is the first layer)
    :param n_filters: the dimensionality of the output space; i.e. the number of output filters
    :param kernel_size: the height and width of the convolution window
    :param dropout: the dropout rate

    :returns: tf.Tensor: The activations of the layer

    """
    # Options for the convolution
    conv_kw = {
        "activation": "relu",
        "kernel_initializer": "he_normal",
        "padding": "same",
    }

    # First convolution layer
    conv = Conv2D(
        n_filters,
        kernel_size,
        **conv_kw,
    )(input_layer)

    # Dropout layer helps prevent overfitting
    conv = Dropout(dropout)(conv)

    # Second convolution layer
    return Conv2D(n_filters, kernel_size, **conv_kw)(conv)


def my_unet(
    n_classes=4,
    IMG_HEIGHT=256,
    IMG_WIDTH=256,
    IMG_CHANNELS=1,
    *,
    dropout: Union[float, list[float]] = 0.1,
):
    """
    Inputs must be normalised

    :param dropout: either the dropout rate for all layers, or for each layer

    """
    n_layers = 9
    if isinstance(dropout, float):
        dropout = [dropout] * n_layers
    else:
        assert len(dropout) == n_layers

    # Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    # inputs = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand

    pooling_size = (2, 2)

    # Contraction path
    c1 = conv_block(inputs, n_filters=16, kernel_size=(3, 3), dropout=dropout[0])
    p1 = MaxPooling2D(pooling_size)(c1)

    c2 = conv_block(p1, n_filters=32, kernel_size=(3, 3), dropout=dropout[1])
    p2 = MaxPooling2D(pooling_size)(c2)

    c3 = conv_block(p2, n_filters=64, kernel_size=(3, 3), dropout=dropout[2])
    p3 = MaxPooling2D(pooling_size)(c3)

    c4 = conv_block(p3, n_filters=128, kernel_size=(3, 3), dropout=dropout[3])
    p4 = MaxPooling2D(pooling_size)(c4)

    # No max pooling for the last layer
    c5 = conv_block(p4, n_filters=256, kernel_size=(3, 3), dropout=dropout[4])

    # Expansive path
    expansive_kw = {"kernel_size": (2, 2), "strides": (2, 2), "padding": "same"}
    u6 = Conv2DTranspose(128, **expansive_kw)(c5)
    u6 = concatenate([u6, c4])
    c6 = conv_block(u6, n_filters=128, kernel_size=(3, 3), dropout=dropout[5])

    u7 = Conv2DTranspose(64, **expansive_kw)(c6)
    u7 = concatenate([u7, c3])
    c7 = conv_block(u7, n_filters=64, kernel_size=(3, 3), dropout=dropout[6])

    u8 = Conv2DTranspose(32, **expansive_kw)(c7)
    u8 = concatenate([u8, c2])
    c8 = conv_block(u8, n_filters=32, kernel_size=(3, 3), dropout=dropout[7])

    u9 = Conv2DTranspose(16, **expansive_kw)(c8)
    u9 = concatenate([u9, c1])
    c9 = conv_block(u9, n_filters=32, kernel_size=(3, 3), dropout=dropout[8])

    outputs = Conv2D(n_classes, (1, 1), activation="softmax")(c9)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model
