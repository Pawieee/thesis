from modules.cbam import cbam_block
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
    shape = x.shape
    filter_count = int(shape[bn_axis] * reduction)

    x = layers.Conv2D(filter_count, 1, use_bias=False, name=name + "_conv")(x)
    x = layers.AveragePooling2D(2, strides=2, name=name + "_pool")(x)
    return x


# ==========================================
# MAIN MODEL BUILDER
# ==========================================


def DenseNetImageNet121(
    input_shape=(224, 224, 3),
    classes=1000,
    include_top=True,
    weights=None,
    attention_module=None,
):
    """
    Constructs a DenseNet-121 model with optional CBAM attention modules.

    Arguments:
        input_shape: tuple, default (224, 224, 3).
        classes: int, number of classes for classification.
        include_top: bool, whether to include the fully-connected layer at the top.
        weights: (Unused in scratch training, but kept for compatibility).
        attention_module: str, 'cbam_block' to enable CBAM, or None for standard DenseNet.
    """
    img_input = layers.Input(shape=input_shape)
    bn_axis = 3 if K.image_data_format() == "channels_last" else 1

    # --- Initial Convolution ---
    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=False, name="conv1_conv")(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name="conv1_bn")(x)
    x = layers.Activation("relu", name="conv1_relu")(x)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.MaxPooling2D(3, strides=2, name="pool1")(x)

    # --- Dense Block 1 (6 layers) ---
    x = dense_block(x, 6, name="conv2")
    if attention_module == "cbam_block":
        x = cbam_block(x, ratio=8)
    x = transition_block(x, 0.5, name="pool2")

    # --- Dense Block 2 (12 layers) ---
    x = dense_block(x, 12, name="conv3")
    if attention_module == "cbam_block":
        x = cbam_block(x, ratio=8)
    x = transition_block(x, 0.5, name="pool3")

    # --- Dense Block 3 (24 layers) ---
    x = dense_block(x, 24, name="conv4")
    if attention_module == "cbam_block":
        x = cbam_block(x, ratio=8)
    x = transition_block(x, 0.5, name="pool4")

    # --- Dense Block 4 (16 layers) ---
    x = dense_block(x, 16, name="conv5")
    if attention_module == "cbam_block":
        x = cbam_block(x, ratio=8)

    # --- Final Layers ---
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name="bn")(x)
    x = layers.Activation("relu", name="relu")(x)

    # If include_top=False, we return the feature map (or global pool it outside)
    if include_top:
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        x = layers.Dense(classes, activation="softmax", name="predictions")(x)

    model_name = (
        "densenet121_cbam" if attention_module == "cbam_block" else "densenet121"
    )
    model = models.Model(img_input, x, name=model_name)

    return model
