import tensorflow as tf

def image_contrast(images, lower=0.2, upper=1.2):
    """
    One technique of data augmentation
    Adjust the constrast of an image by a random factor,
    picked in a specified interval [lower, upper],

    """
    images = tf.reshape(images, [-1, 32, 32, 3])
    distorted_image = tf.map_fn(lambda img: tf.image.random_contrast(img, lower=lower, upper=upper), images)
    distorted_image = tf.reshape(distorted_image, [-1, 32*32*3])
    return distorted_image
