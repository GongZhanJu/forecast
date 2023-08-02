import tensorflow as tf


def conv_layer(name, input, w_shape, b_shape, stride):
    with tf.variable_scope(name) as scope:
        weights = tf.Variable(tf.truncated_normal(shape=w_shape, stddev=1.0, dtype=tf.float32),
                              name='weights', dtype=tf.float32)
        biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=b_shape),
                             name='biases', dtype=tf.float32)
        conv = tf.nn.conv2d(input, weights, strides=[1, stride[0], stride[1], 1], padding='SAME')
        conv = tf.nn.bias_add(conv, biases)
        conv_out = tf.nn.relu(conv)
        return conv_out


def max_pool_lrn(name, input, ksize, is_lrn):
    with tf.variable_scope(name) as scope:
        max_pool_out = tf.nn.max_pool(input, ksize=ksize, strides=[1, 2, 2, 1], padding='SAME')
        if is_lrn:
            max_pool_out = tf.nn.lrn(max_pool_out, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    return max_pool_out


def res_block(name, num_blocks, input, inchannel, outchannel, stride):
    conv_input = input
    conv_inchannel = inchannel
    conv_stride = stride
    for i in range(num_blocks):
        out = conv_layer(name='{}_{}'.format(name, i), input=conv_input,
                         w_shape=[3, 3, conv_inchannel, outchannel],
                         b_shape=[outchannel], stride=[conv_stride, conv_stride])
        conv_input = out
        conv_inchannel = outchannel
        conv_stride = 1

    if stride > 1:
        shortcut = conv_layer(name='{}_{}'.format(name, i), input=input,
                              w_shape=[1, 1, inchannel, outchannel],
                              b_shape=[outchannel], stride=[stride, stride])
        out = out + shortcut
    return out


#def inference(images, batch_size, drop_rate):
#    pre_conv1 = conv_layer('pre_conv', images, [7, 7, 3, 64], [64], [2, 2])
#    pool_1 = max_pool_lrn('pooling1', pre_conv1, [1, 3, 3, 1], is_lrn=False)
#    layer1 = res_block('Resblock1', 2, pool_1, 64, 64, 1)
#    layer2 = res_block('Resblock2', 2, layer1, 64, 128, 2)
#    layer3 = res_block('Resblock3', 2, layer2, 128, 256, 2)
#   layer4 = res_block('Resblock4', 2, layer3, 256, 512, 2)
#    global_avg = tf.nn.avg_pool(layer4, ksize=[1, 7, 7, 1], strides=[1, 7, 7, 1], padding='SAME')
#
#    reshape = tf.reshape(global_avg, shape=[batch_size, -1])
#    dim = reshape.get_shape()[1].value
#
#    with tf.variable_scope('softmax_linear') as scope:
#        weights = tf.Variable(tf.truncated_normal(shape=[dim, 1], stddev=0.005, dtype=tf.float32),
#                              name='softmax_linear', dtype=tf.float32)
#        biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[1]),
#                             name='biases', dtype=tf.float32)
#        logits = tf.add(tf.matmul(reshape, weights), biases, name='softmax_linear')
#    return tf.squeeze(logits)
def inference(images, batch_size, drop_rate=0.5):
    pre_conv1 = conv_layer('pre_conv', images, [7, 7, 3, 64], [64], [2, 2])
    pool_1 = max_pool_lrn('pooling1', pre_conv1, [1, 3, 3, 1], is_lrn=False)
    layer1 = res_block('Resblock1', 2, pool_1, 64, 64, 1)
    layer2 = res_block('Resblock2', 2, layer1, 64, 128, 2)
    layer3 = res_block('Resblock3', 2, layer2, 128, 256, 2)
    layer4 = res_block('Resblock4', 2, layer3, 256, 512, 2)
    global_avg = tf.nn.avg_pool(layer4, ksize=[1, 7, 7, 1], strides=[1, 7, 7, 1], padding='SAME')

    reshape = tf.reshape(global_avg, shape=[batch_size, -1])
    dim = reshape.get_shape()[1].value

    dropout = tf.nn.dropout(reshape, rate=drop_rate)

    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.Variable(tf.truncated_normal(shape=[dim, 1], stddev=0.005, dtype=tf.float32),
                              name='softmax_linear', dtype=tf.float32)
        biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[1]),
                             name='biases', dtype=tf.float32)
        logits = tf.add(tf.matmul(dropout, weights), biases, name='softmax_linear')
    return tf.squeeze(logits)


def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        labels = tf.cast(labels, tf.float32)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
        tf.summary.scalar(scope.name + '/loss', loss)
    return loss


def trainning(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        predictions = tf.cast(tf.greater_equal(logits, 0.5), tf.float32)
        correct = tf.equal(predictions, labels)
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy