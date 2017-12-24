import tensorflow as tf
import numpy as np
def image_flip_brightness(images, max_delta=63/255.0):
    """
    One technique of data augmentation
    Ajust the brightness of image by a random factor picked in
    a specified interval [-max_delta, max_delta],
    this is done after flipping

    """
    images = tf.reshape(images, [-1, 32, 32, 3])
    distorted_image = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), images)
    distorted_image = tf.map_fn(lambda img: tf.image.random_brightness(img, max_delta=max_delta), distorted_image)
    distorted_image = tf.reshape(distorted_image, [-1, 32*32*3])
    return distorted_image
