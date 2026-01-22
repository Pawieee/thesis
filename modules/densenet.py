from modules.cbam import cbam_block
import tensorflow as tf

from keras import layers, models, backend as K


def conv_block(x, growth_rate, name):
    """
    A building block for a dense block.
    BN -> ReLU -> Conv1x1 -> BN -> ReLU -> Conv3x3
    """
    bn_axis = 3 if K.image_data_format() == "channels_last" else 1

    # 1x1 Convolution (Bottleneck layer)
    x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_0_bn")(
        x
    )
    x1 = layers.Activation("relu", name=name + "_0_relu")(x1)
    x1 = layers.Conv2D(4 * growth_rate, 1, use_bias=False, name=name + "_1_conv")(x1)

    # 3x3 Convolution
    x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_1_bn")(
        x1
    )
    x1 = layers.Activation("relu", name=name + "_1_relu")(x1)
    x1 = layers.Conv2D(
        growth_rate, 3, padding="same", use_bias=False, name=name + "_2_conv"
    )(x1)

    # Concatenate input with output
    x = layers.Concatenate(axis=bn_axis, name=name + "_concat")([x, x1])
    return x


def dense_block(x, blocks, name):
    """
    A dense block consists of `blocks` number of conv_blocks.
    """
    for i in range(blocks):
        x = conv_block(x, 32, name=name + "_block" + str(i + 1))
    return x


def transition_block(x, reduction, name):
    """
    A transition block to reduce feature map size.
    BN -> ReLU -> Conv1x1 -> AvgPool2D
    """
    bn_axis = 3 if K.image_data_format() == "channels_last" else 1

    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_bn")(x)
    x = layers.Activation("relu", name=name + "_relu")(x)

    # Compress number of channels
    # FIX: Use .shape instead of K.int_shape(x) for Keras 3 compatibility
    shape = x.shape
    filter_count = int(shape[bn_axis] * reduction)

    x = layers.Conv2D(filter_count, 1, use_bias=False, name=name + "_conv")(x)
    x = layers.AveragePooling2D(2, strides=2, name=name + "_pool")(x)
    return x


# ==========================================
# MAIN MODEL BUILDER
# ==========================================


def DenseNet121_CBAM(input_shape=(224, 224, 3), classes=1000, include_top=True):
    """
    Constructs a DenseNet-121 model with CBAM attention modules.

    Arguments:
        input_shape: tuple, default (224, 224, 3).
        classes: int, number of classes for classification.
        include_top: bool, whether to include the fully-connected layer at the top of the network.

    Returns:
        A tf.keras.Model instance.
    """
    img_input = layers.Input(shape=input_shape)
    bn_axis = 3 if K.image_data_format() == "channels_last" else 1

    # --- Initial Convolution ---
    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)

    # FIX: Renamed 'conv1/conv' to 'conv1_conv' to allow compilation in Keras 3+
    # Keras 3 strict naming rules do not allow '/' in layer names.
    x = layers.Conv2D(64, 7, strides=2, use_bias=False, name="conv1_conv")(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name="conv1_bn")(x)
    x = layers.Activation("relu", name="conv1_relu")(x)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.MaxPooling2D(3, strides=2, name="pool1")(x)

    # --- Dense Block 1 (6 layers) + CBAM ---
    x = dense_block(x, 6, name="conv2")
    x = cbam_block(x, ratio=8)  # Inject CBAM
    x = transition_block(x, 0.5, name="pool2")

    # --- Dense Block 2 (12 layers) + CBAM ---
    x = dense_block(x, 12, name="conv3")
    x = cbam_block(x, ratio=8)  # Inject CBAM
    x = transition_block(x, 0.5, name="pool3")

    # --- Dense Block 3 (24 layers) + CBAM ---
    x = dense_block(x, 24, name="conv4")
    x = cbam_block(x, ratio=8)  # Inject CBAM
    x = transition_block(x, 0.5, name="pool4")

    # --- Dense Block 4 (16 layers) + CBAM ---
    x = dense_block(x, 16, name="conv5")
    x = cbam_block(x, ratio=8)  # Inject CBAM

    # --- Final Classification Layer ---
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name="bn")(x)
    x = layers.Activation("relu", name="relu")(x)

    # Use global average pooling for feature extraction
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)

    # Optional Classification Head
    if include_top:
        x = layers.Dense(classes, activation="softmax", name="predictions")(x)

    model = models.Model(img_input, x, name="densenet121_cbam")
    return model


# ==========================================
# EXAMPLE USAGE & TEST
# ==========================================
if __name__ == "__main__":
    print(f"TensorFlow Version: {tf.__version__}")

    try:
        model = DenseNet121_CBAM(
            input_shape=(224, 224, 3), classes=55, include_top=True
        )
        model.summary()
        print("\nSUCCESS: Model created and compiled successfully!")

        # Test basic inference with random data
        import numpy as np

        dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
        output = model.predict(dummy_input)
        print(f" SUCCESS: Inference check passed. Output shape: {output.shape}")

    except Exception as e:
        print(f"\nERROR: {e}")
