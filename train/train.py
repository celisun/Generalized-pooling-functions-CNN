import os
import datetime
import numpy as np
import tensorflow as tf
import mlp.data_providers as data_providers

from image_processing import *




os.environ['MLP_DATA_DIR'] = '/disk/scratch/mlp/data'
os.environ['OUTPUT_DIR'] = '$HOME/experiments'


# check necessary environment variables are defined
assert 'MLP_DATA_DIR' in os.environ, (
    'An environment variable MLP_DATA_DIR must be set to the path containing'
    ' MLP data before running script.')
assert 'OUTPUT_DIR' in os.environ, (
    'An environment variable OUTPUT_DIR must be set to the path to write'
    ' output to before running script.')



# load data
train_data = data_providers.CIFAR10DataProvider('train', batch_size=100)
valid_data = data_providers.CIFAR10DataProvider('valid', batch_size=100)
valid_inputs = valid_data.inputs
valid_targets = valid_data.to_one_of_k(valid_data.targets)




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




# Convolutional layer with non-linearity
def conv2d(x, w, b, activation=tf.nn.relu):
    conv = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')
    return activation(tf.nn.bias_add(conv, b))


# fully connected layer with non-linearity
def fc(x, w, b, activation=tf.nn.relu):
    y = tf.nn.bias_add (tf.matmul(x, w), b)
    if(activation != 'None'):
        y = activation(y)
    return y





with tf.name_scope('data'):
    inputs = tf.placeholder(tf.float32, [None, 3072], name='inputs')
    targets = tf.placeholder(tf.float32, [None, train_data.num_classes], name='targets')
with tf.name_scope('parameters'):
    weights = {
        'W_conv1':tf.Variable(tf.truncated_normal([3,3,3,32], stddev=2./(9*3+9*32)**0.5)),
        'W_conv2':tf.Variable(tf.truncated_normal([3,3,32,64], stddev=2./(9*32+9*64)**0.5)),
        'W_conv3':tf.Variable(tf.truncated_normal([3,3,64,128], stddev=2./(9*64+9*128)**0.5)),
        'W_conv4':tf.Variable(tf.truncated_normal([3,3,128,128], stddev=2./(9*128*2)**0.5)),
        'W_fc1':tf.Variable(tf.truncated_normal([8*8*128, 1024], stddev=2./(8*8*128+1024)**0.5)),
        'W_fc2':tf.Variable(tf.truncated_normal([1024, 512], stddev=2./(1024+512)**0.5)),
        'W_fc3':tf.Variable(tf.truncated_normal([512, train_data.num_classes], stddev=2./(train_data.num_classes+512)**0.5))
    }
    biases = {
        'b_conv1':tf.Variable(tf.zeros([32])),
        'b_conv2':tf.Variable(tf.zeros([64])),
        'b_conv3':tf.Variable(tf.zeros([128])),
        'b_conv4':tf.Variable(tf.zeros([128])),
        'b_fc1':tf.Variable(tf.zeros([1024])),
        'b_fc2':tf.Variable(tf.zeros([512])),
        'b_fc3':tf.Variable(tf.zeros([train_data.num_classes]))
    }
# specify if apply data augmentation to input
if_augmentation = True

with tf.name_scope('model'):
    if(if_augmentation):
        inputs = image_cropping(inputs)
    input_layer = tf.reshape(inputs, [-1,32,32,3])
    # Convolutional layer   #1 #2
    conv1 = conv2d(
        input_layer, weights['W_conv1'], biases['b_conv1'])
    conv2 = conv2d(
        conv1, weights['W_conv2'], biases['b_conv2'])
    # Pooling Layer #1
    #pool1 = tf.contrib.layers.max_pool2d(inputs=conv2, kernel_size=[2,2],stride=2, padding='VALID')
    #pool1 = tf.contrib.layers.avg_pool2d(inputs=conv2, kernel_size=[2, 2],stride=2, padding='VALID')
    alpha, pool1 = mixed_pooling(conv2, alpha=-1)
    #pool1 = gated_pooling(conv2, filter=64)
    #pool1 = gated_pooling(conv2, filter=64, learn_option='l')
    #pool1 = gated_pooling(conv2, filter=64, learn_option='l/r/c')

    # Convolutional layer   #3 #4
    conv3 = conv2d(
        pool1,weights['W_conv3'],biases['b_conv3'])
    conv4 = conv2d(
        conv3, weights['W_conv4'],biases['b_conv4'])
    # Pooling Lyaer #2
    #pool2 = tf.contrib.layers.max_pool2d(inputs=conv4,kernel_size=[2,2],stride=2, padding='VALID')
    #pool2 = tf.contrib.layers.avg_pool2d(inputs=conv4,kernel_size=[2,2],stride=2, padding='VALID')
    _, pool2 = mixed_pooling(conv4, alpha=alpha)
    #_, pool2 = mixed_pooling(conv4, alpha=-1)
    #pool2 = gated_pooling(conv4, filter=128)
    #pool2 = gated_pooling(conv4, filter=128, learn_option='l')
    #pool2 = gated_pooling(conv4, filter=128, learn_option='l/r/c')
    pool2_flat = tf.reshape(pool2, [-1, 8*8*128])

    fc1= fc(pool2_flat, weights['W_fc1'], biases['b_fc1'])
    dropout1 = tf.nn.dropout(fc1, 0.4)
    fc2= fc(dropout1,  weights['W_fc2'], biases['b_fc2'])
    fc3= fc(fc2,  weights['W_fc3'], biases['b_fc3'], activation = 'None')
    outputs = fc3



# Anneal to learning rate
global_step= tf.Variable(0, trainable=False)
boundaries = [20, 40, 60, 80]
values = [0.01, 0.005, 0.001, 0.0005, 0.0001]
learning_rate1 = tf.train.piecewise_constant(
    global_step,boundaries , values)
learning_rate2 = tf.train.exponential_decay(
        learning_rate=0.5, global_step=global_step,
        decay_steps=100, decay_rate=0.001)
weight_decay = tf.nn.l2_loss(weights['W_conv1'])+tf.nn.l2_loss(weights['W_conv2']) + tf.nn.l2_loss(weights['W_conv3'])\
               + tf.nn.l2_loss(weights['W_conv4'])+tf.nn.l2_loss(weights['W_fc1'])+tf.nn.l2_loss(weights['W_fc2'])+\
               tf.nn.l2_loss(weights['W_fc3'])






with tf.name_scope('error'):
    error = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(outputs, targets) + 0.001*weight_decay)
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(
        tf.argmax(outputs, 1), tf.argmax(targets, 1)), tf.float32))
with tf.name_scope('train'):
    #train_step = tf.train.AdagradOptimizer(learning_rate=0.01).minimize(error, global_step=global_step)
    train_step = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9).minimize(error, global_step=global_step)
    #train_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(error)



# ---------------------------------------------------------------------------------


# add summary operations
tf.summary.scalar('error', error)
tf.summary.scalar('accuracy', accuracy)
summary_op = tf.summary.merge_all()
# create objects for writing summaries and checkpoints during training
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
exp_dir = os.path.join(os.environ['OUTPUT_DIR'], timestamp)
checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
train_writer = tf.summary.FileWriter(os.path.join(exp_dir, 'train-summaries'))
valid_writer = tf.summary.FileWriter(os.path.join(exp_dir, 'valid-summaries'))
saver = tf.train.Saver()


# create arrays to store run train / valid set stats
num_epoch = 75
train_accuracy = np.zeros(num_epoch)
train_error = np.zeros(num_epoch)
valid_accuracy = np.zeros(num_epoch)
valid_error = np.zeros(num_epoch)






# create session and run training loop
#sess = tf.Session()
NUM_THREADS = 8
sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS))
sess.run(tf.global_variables_initializer())
step = 0
print('starting....10+')
for e in range(num_epoch):
    for b, (input_batch, target_batch) in enumerate(train_data):
        # do train step with current batch
        _, summary, batch_error, batch_acc = sess.run(
            [train_step, summary_op, error, accuracy],
            feed_dict={
                inputs: input_batch,
                targets: target_batch
            })
        # add symmary and accumulate stats
        train_writer.add_summary(summary, step)
        train_error[e] += batch_error
        train_accuracy[e] += batch_acc
        step += 1
    # normalise running means by number of batches
    train_error[e] /= train_data.num_batches
    train_accuracy[e] /= train_data.num_batches
    # evaluate validation set performance
    valid_summary, valid_error[e], valid_accuracy[e] = sess.run(
        [summary_op, error, accuracy],
        feed_dict={inputs: valid_inputs, targets: valid_targets})
    valid_writer.add_summary(valid_summary, step)
    # checkpoint model variables
    saver.save(sess, os.path.join(checkpoint_dir, 'model.ckpt'), step)
    # write stats summary to stdout
    print('Epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'
          .format(e + 1, train_error[e], train_accuracy[e]))
    print('          err(valid)={0:.4f} acc(valid)={1:.4f}'
          .format(valid_error[e], valid_accuracy[e]))


# close writer and session objects
train_writer.close()
valid_writer.close()
sess.close()












