from __future__ import absolute_import, division, print_function
from typing import Tuple

import tensorflow as tf

def mobile_net(label_names, pic_size=224, channels=3, image_batch=None, verbose=False):
    mobile_net = tf.keras.applications.MobileNetV2(input_shape=(pic_size, pic_size, channels), include_top=False)

    if verbose and not image_batch:
        print("Please provide 'image_batch' param for verbose output")

    if verbose and image_batch:
        feature_map_batch = mobile_net(image_batch)
        print(feature_map_batch.shape)

    model = tf.keras.Sequential([
        mobile_net,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(label_names))])

    return model