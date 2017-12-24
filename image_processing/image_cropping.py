import tensorflow as tf
 

def image_cropping(images, central_fraction=0.75):
    """
    One technique of data augmentation, applies to input tensor,
     cropping the central area of origin 32x32 CIFAR images by a certain fraction.
     image is then re-sized to origin 32x32

      argument:
        images: input tensor of shape [batch_size, 3072]
      return:
        distorted_image: tensor of the same shape as input

    """
    images = tf.reshape(images, [-1, 32, 32, 3])
    distorted_image = tf.map_fn(lambda img: tf.image.central_crop(img, central_fraction), images)
    distorted_image = tf.image.resize_images(distorted_image, [32, 32])
    distorted_image = tf.reshape(distorted_image, [-1, 32*32*3])
    return distorted_image
