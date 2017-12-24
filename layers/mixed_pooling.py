import tensorflow as tf

# define mixed max-average pooling layer
def mixed_pooling (inputs, alpha, size=2):
    """Mixed pooling operation, nonresponsive
       Combine max pooling and average pooling in fixed proportion specified by alpha a:
        f mixed (x) = a * f max(x) + (1-a) * f avg(x)

        arguments:
          inputs: tensor of shape [batch size, height, width, channels]
          size: an integer, width and height of the pooling filter
          alpha: the scalar mixing proportion of range [0,1]
        return:
          outputs: tensor of shape [batch_size, height//size, width//size, channels]

    """
    if alpha == -1:
        alpha = tf.Variable(0.0)
    x1 = tf.contrib.layers.max_pool2d(inputs=inputs, kernel_size=[size, size], stride=2, padding='VALID')
    x2 = tf.contrib.layers.avg_pool2d(inputs=inputs, kernel_size=[size, size], stride=2, padding='VALID')
    outputs = tf.add(tf.multiply(x1, alpha), tf.multiply(x2, (1-alpha)))

    return [alpha, outputs]




