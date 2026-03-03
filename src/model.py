import tensorflow as tf
from tensorflow.keras import layers, models


def build_eegnet(n_channels: int, n_times: int, dropout_rate: float = 0.3):
    """
    Build EEGNet-based binary classification model.
    """

    input1 = layers.Input(shape=(n_channels, n_times, 1))

    # Temporal convolution
    x = layers.Conv2D(16, (1, 64), padding='same', use_bias=False)(input1)
    x = layers.BatchNormalization()(x)

    # Depthwise spatial convolution
    x = layers.DepthwiseConv2D(
        (n_channels, 1),
        use_bias=False,
        depth_multiplier=2,
        depthwise_constraint=tf.keras.constraints.max_norm(1.)
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = layers.AveragePooling2D((1, 4))(x)
    x = layers.Dropout(dropout_rate)(x)

    # Separable convolution
    x = layers.SeparableConv2D(
        32,
        (1, 16),
        use_bias=False,
        padding='same'
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = layers.AveragePooling2D((1, 8))(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Flatten()(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    return models.Model(inputs=input1, outputs=output)