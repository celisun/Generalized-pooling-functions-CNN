
import tensorflow as tf


# define gated max-average pooling lyaer
def gated_pooling(inputs, filter, size=2, learn_option='l/c'):
    """Gated pooling operation, responsive
    Combine max pooling and average pooling in a mixing proportion,
    which is obtained from the inner product between the gating mask and the region being
    pooled and then fed through a sigmoid:
       fgate(x) =  sigmoid(w*x)* fmax(x) + (1-sigmoid(w*x))* favg(x)

       arguments:
         inputs: input of shape [batch size, height, width, channels]
         filter: filter size of the input layer, used to initialize gating mask
         size: an integer, width and height of the pooling filter
         learn_option: learning options of gated pooling, include:
                        'l/c': learn a mask per layer/channel
                        'l/r/c': learn a mask per layer/pooling region/channel combined
       return:
         outputs: tensor with the shape of [batch_size, height//size, width//size, channels]

    """
    if learn_option == 'l':
        gating_mask = all_channel_connected2d(inputs)
    if learn_option == 'l/c':
        w_gated = tf.Variable(tf.truncated_normal([size,size,filter,filter], stddev=2/(size*size*filter*2)**0.5))
        gating_mask = tf.nn.conv2d(inputs, w_gated, strides=[1,size,size,1], padding='VALID')
    if learn_option == 'l/r/c':
        gating_mask = locally_connected2d(inputs)

    alpha = tf.sigmoid(gating_mask)

    x1 = tf.contrib.layers.max_pool2d(inputs=inputs, kernel_size=[size, size], stride=2, padding='VALID')
    x2 = tf.contrib.layers.avg_pool2d(inputs=inputs, kernel_size=[size, size],stride=2, padding='VALID')
    outputs = tf.add(tf.multiply(x1, alpha), tf.multiply(x2, (1-alpha)))
    return outputs





#locally connected layer (unshared-weights conv, layer),
# designed for gated pooling, learn a param "per layer/region/channel"
def locally_connected2d(x, size = 2):
    """
    The `LocallyConnected2D` layer works similarly
    to the `Convolution2D` layer, except that weights are unshared,
    that is, a different set of filters is applied at each
    different patch of the input.

    NOTE: No bias or activation function applied. No overlapping between sub-region.

    arguments:
        x: 4D tensor with shape: [samples, rows, cols, channels]
        size: width and height of the filter, default 2x2 filter.
              this is also the length of stride to ensure no overlapping
    returns:
        4D tensor with shape: [samples, new_rows, new_cols, nb_filter]
        `rows` and `cols` values might have changed due to padding.

    """

    xs = []
    _, input_row, input_col, nb_filter = x.get_shape().as_list()
    output_row = input_row //2
    output_col = input_col //2
    nb_row = size
    nb_col = size
    stride_row = size
    stride_col = size
    feature_dim = nb_row * nb_col * nb_filter

    w_shape = (output_row * output_col,
               nb_row * nb_col * nb_filter,
               nb_filter)
    mask = tf.Variable(tf.truncated_normal(w_shape, stddev=2./(w_shape[0]*w_shape[1]*2)**0.5))
    for i in range(output_row):
        for j in range(output_col):
            slice_row = slice(i * stride_row,
                              i * stride_row + nb_row)
            slice_col = slice(j * stride_col,
                              j * stride_col + nb_col)
            xs.append(tf.reshape(x[:, slice_row, slice_col, :], (1, -1, feature_dim)))
    x_aggregate = tf.concat(0, xs)
    output = tf.matmul(x_aggregate, mask)
    output = tf.reshape(output, (output_row, output_col, -1, nb_filter))
    output = tf.transpose(output, perm=[2, 0, 1, 3])

    return output



#design for gated pooling, learn a param "per layer" option
def all_channel_connected2d(x, size=2):
    """
    The all channel connected layer is a modified version of
    Convolutional layer,
    which shares the same weights not only between each patch,
    but also between all channels of the layer input. That is,
    the whole layer only has one filter

    NOTE: 'VALID', no bias, no activation function.

    arguments:
        x: 4D tensor with shape: [batch_size, rows, cols, channels]
        size: width and height of the filter, default 2x2 filter.
              this is also the length of stride to ensure no overlapping
    returns:
        4D tensor with shape: [batch_size, new_rows, new_cols, nb_filter]
    """

    nb_batch, input_row, input_col, nb_filter = x.get_shape().as_list()
    output_size = input_row //2
    mask = tf.Variable(tf.truncated_normal([size,size,1,1], stddev=2./(size*size*2)**0.5))

    xs = []
    for c in tf.split(x, nb_filter, 3):
        xs.append(tf.nn.conv2d(c, mask, strides=[1,1,1,1], padding='VALID'))
    output = tf.reshape(x, [nb_batch, output_size, output_size, nb_filter])

    return output
