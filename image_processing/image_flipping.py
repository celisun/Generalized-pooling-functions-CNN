import tensorflow as tf
 
def image_flipping(images):
    """
    One technique of data augmentation
    randomly flip the image horizontally from left to right
    """
    images = tf.reshape(images, [-1, 32, 32, 3])
    distorted_image = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), images)
    distorted_image = tf.reshape(distorted_image, [-1, 32*32*3])
    return distorted_image


