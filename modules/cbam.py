import tensorflow as tf

from keras import layers, backend as K

# Utility for generating unique IDs without relying on K.get_uid
_uid_counter = 0


def get_unique_id(prefix):
    global _uid_counter
    _uid_counter += 1
    return f"{prefix}_{_uid_counter}"


def channel_attention(input_feature, ratio=8):
    """
    CBAM Channel Attention Module
    """
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    # Get channel count (handle dynamic shapes if necessary, though dense usually static)
    channel = input_feature.shape[channel_axis]

    # If channel is None (dynamic), we might need to handle it, but typically
    # DenseNet has known shapes at build time.
    # For safety in some TF versions, ensure channel is an int if possible.
    if channel is None:
        raise ValueError(
            "Channel dimension of the inputs should be defined. Found `None`."
        )

    # Shared MLP
    shared_layer_one = layers.Dense(
        channel // ratio,
        activation="relu",
        kernel_initializer="he_normal",
        use_bias=True,
        bias_initializer="zeros",
    )
    shared_layer_two = layers.Dense(
        channel, kernel_initializer="he_normal", use_bias=True, bias_initializer="zeros"
    )

    # Global Average Pooling
    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    # Global Max Pooling
    max_pool = layers.GlobalMaxPooling2D()(input_feature)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    # Add and Activate
    cbam_feature = layers.Add()([avg_pool, max_pool])
    cbam_feature = layers.Activation("sigmoid")(cbam_feature)

    return layers.Multiply()([input_feature, cbam_feature])


def spatial_attention(input_feature):
    """
    CBAM Spatial Attention Module
    Uses Lambda layers to wrap raw TensorFlow operations for Keras 3+ compatibility.
    """
    kernel_size = 7

    # Determine axis based on data format
    if K.image_data_format() == "channels_first":
        channel_axis = 1
    else:
        channel_axis = -1  # channels_last

    # WRAPPER FIX: Wrap raw TF ops in Lambda layers so they work in Functional API
    # We use a custom name generator to avoid K.get_uid issues
    avg_pool_name = get_unique_id("spatial_att_avg")
    max_pool_name = get_unique_id("spatial_att_max")

    avg_pool = layers.Lambda(
        lambda x: tf.reduce_mean(x, axis=channel_axis, keepdims=True),
        name=avg_pool_name,
    )(input_feature)
    max_pool = layers.Lambda(
        lambda x: tf.reduce_max(x, axis=channel_axis, keepdims=True), name=max_pool_name
    )(input_feature)

    concat = layers.Concatenate(axis=channel_axis)([avg_pool, max_pool])

    cbam_feature = layers.Conv2D(
        filters=1,
        kernel_size=kernel_size,
        strides=1,
        padding="same",
        activation="sigmoid",
        kernel_initializer="he_normal",
        use_bias=False,
    )(concat)

    return layers.Multiply()([input_feature, cbam_feature])


def cbam_block(input_feature, ratio=8):
    """
    Combined CBAM Block (Channel + Spatial)
    """
    x = channel_attention(input_feature, ratio)
    x = spatial_attention(x)
    return x
