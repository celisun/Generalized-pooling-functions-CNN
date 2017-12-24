
def image_rot90(images):
    """
    One technique of data augmentation
    rotate all images by 90 degree
    """
    images = tf.reshape(images, [-1, 32, 32, 3])
    distorted_image = tf.map_fn(lambda img: tf.image.rot90(img), images)
    distorted_image = tf.reshape(distorted_image, [-1, 32*32*3])
    return distorted_image

