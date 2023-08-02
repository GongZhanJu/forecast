# -*- coding:utf-8 -*-
# =========================================================================
import tensorflow as tf


# =========================================================================
# 网络结构定义
# 输入参数：images，image batch、4D tensor、tf.float32、[batch_size, width, height, channels]
# 返回参数：logits, float、 [batch_size, n_classes]

# 一个简单的卷积神经网络，卷积+池化层x2，全连接层x2，最后一个softmax层做分类。
# 卷积层1
# 64个3x3的卷积核（3通道），padding=’SAME’，表示padding后卷积的图与原图尺寸一致，激活函数relu()
# with tf.variable_scope('conv1') as scope:
#     weights = tf.Variable(tf.truncated_normal(shape=[2, 2, 3, 64], stddev=1.0, dtype=tf.float32),
#                           name='weights', dtype=tf.float32)

#     biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[64]),
#                          name='biases', dtype=tf.float32)

#     conv = tf.nn.conv2d(images, weights, strides=[1, 2, 2, 1], padding='SAME')
#     pre_activation = tf.nn.bias_add(conv, biases)
#     conv1 = tf.nn.relu(pre_activation, name=scope.name)

def Conv_layer(names, input, w_shape, b_shape, strid):
    with tf.variable_scope(names) as scope:
        weights = tf.Variable(tf.truncated_normal(shape=w_shape, stddev=1.0, dtype=tf.float32),
                              name='weights_{}'.format(names), dtype=tf.float32)

        biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=b_shape),
                             name='biases_{}'.format(names), dtype=tf.float32)

        conv = tf.nn.conv2d(input, weights, strides=[1, strid[0], strid[1], 1], padding='SAME')
        # print(strid)
        conv = tf.nn.bias_add(conv, biases)
        conv_out = tf.nn.relu(conv, name='relu_{}'.format(names))

        # print("---------names:{}".format(conv_out))
        return conv_out


def Max_pool_lrn(names, input, ksize, is_lrn):
    with tf.variable_scope(names) as scope:
        Max_pool_out = tf.nn.max_pool(input, ksize=ksize, strides=[1, 2, 2, 1], padding='SAME',
                                      name='max_pool_{}'.format(names))
        if is_lrn:
            Max_pool_out = tf.nn.lrn(Max_pool_out, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                                     name='lrn_{}'.format(names))
            # print("use lrn operation")
    return Max_pool_out


def Dropout_layer(names, input, drop_rate):
    with tf.variable_scope(names) as scope:
        # drop_out =local3
        drop_out = tf.nn.dropout(input, drop_rate)
    return drop_out


def local_layer(names, input, w_shape, b_shape):
    with tf.variable_scope(names) as scope:
        weights = tf.Variable(tf.truncated_normal(shape=w_shape, stddev=0.005, dtype=tf.float32),
                              name='weights_{}'.format(names), dtype=tf.float32)

        biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=b_shape),
                             name='biases_{}'.format(names), dtype=tf.float32)

        local = tf.nn.relu(tf.matmul(input, weights) + biases, name='local_{}'.format(names))
    return local


def ResBlock(name, num_blocks, input, inchannel, outchannel, stride):
    conv_input = input
    conv_inchannel = inchannel
    conv_stride = stride
    for i in range(num_blocks):
        out = Conv_layer(names='{}{}'.format(name, i), input=conv_input, w_shape=[3, 3, conv_inchannel, outchannel],
                         b_shape=[outchannel], strid=[conv_stride, conv_stride])
        conv_input = out
        conv_inchannel = outchannel
        conv_stride = 1

    # 残差
    if stride > 1:
        shortcut = Conv_layer(names='{}_{}'.format(name, i), input=input, w_shape=[1, 1, inchannel, outchannel],
                              b_shape=[outchannel], strid=[stride, stride])
        out = out + shortcut
    return out

    # print("******** out {} ".format(out.shape))


# 网络结构

# 预处理层:      输入特征图尺寸:[B,224,224,3], 卷积核尺寸：7x7,   输出通道数：64, 步长：2,    输出特征图大小:[B,112,112,64]

# 最大池化层:    输入特征图尺寸:[B,112,112,64], 池化核尺寸：3x3,   输出通道数：64, 步长：2,    输出特征图大小:[B,56,56,64]

# 残差块1：      输入特征图尺寸:[B,56,56,64],  卷积核尺寸：3x3,   输出通道数：64, 步长：2,    输出特征图大小：[B,56,56,64]

# 残差块2：      输入特征图尺寸:[B,56,56,64],  卷积核尺寸：3x3,   输出通道数：128, 步长：2,   输出特征图大小：[B,28,28,128]

# 残差块3：      输入特征图尺寸:[B,28,28,128], 卷积核尺寸：3x3,   输出通道数：256, 步长：2,   输出特征图大小：[B,14,14,256]

# 残差块4：      输入特征图尺寸:[B,14,14,256], 卷积核尺寸：3x3,   输出通道数：512, 步长：2,   输出特征图大小：[B,7,7,512]

# 全局平均池化层：输入特征图尺寸:[B,7,7,512],   池化核尺寸：7x7,   输出通道数：512,           输出特征图大小：[B,1,1,512]


def inference(images, batch_size, n_classes, drop_rate):
    print("******** images {} ".format(images.shape))
    # 第一层预处理卷积
    pre_conv1 = Conv_layer(names='pre_conv', input=images, w_shape=[7, 7, 3, 64], b_shape=[64], strid=[2, 2])
    print("******** pre_conv1 {} ".format(pre_conv1.shape))

    # 池化层
    pool_1 = Max_pool_lrn(names='pooling1', input=pre_conv1, ksize=[1, 3, 3, 1], is_lrn=False)
    # print("******** pool_1 {} ".format(pool_1.shape))

    # 第一个卷积块(layer1)
    layer1 = ResBlock('Resblock1', 2, pool_1, 64, 64, 1)
    print("******** layer1 {} ".format(layer1.shape))

    # 第二个卷积块(layer2)
    layer2 = ResBlock('Resblock2', 2, layer1, 64, 128, 2)
    print("******** layer2 {} ".format(layer2.shape))

    # 第三个卷积块(layer3)
    layer3 = ResBlock('Resblock3', 2, layer2, 128, 256, 2)
    print("******** layer3 {} ".format(layer3.shape))

    # 第四个卷积块(layer4)
    layer4 = ResBlock('Resblock4', 2, layer3, 256, 512, 2)
    print("******** layer4 {} ".format(layer4.shape))

    # 全局平均池化
    global_avg = tf.nn.avg_pool(layer4, ksize=[1, 7, 7, 1], strides=[1, 7, 7, 1], padding='SAME')
    print("******** global_avg {} ".format(global_avg.shape))

    reshape = tf.reshape(global_avg, shape=[batch_size, -1])
    dim = reshape.get_shape()[1].value

    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.Variable(tf.truncated_normal(shape=[dim, n_classes], stddev=0.005, dtype=tf.float32),
                              name='softmax_linear', dtype=tf.float32)

        biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[n_classes]),
                             name='biases', dtype=tf.float32)

        resnet18_out = tf.add(tf.matmul(reshape, weights), biases, name='softmax_linear')
        print("---------resnet18_out:{}".format(resnet18_out))

    return resnet18_out


# -----------------------------------------------------------------------------
# loss计算
# 传入参数：logits，网络计算输出值。labels，真实值，在这里是0或者1
# 返回参数：loss，损失值


def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                       name='xentropy_per_example')
        # print("\ncross_entropy:{},cross_entropy.shape:{}".format(cross_entropy,cross_entropy.shape))
        # print("---------cross_entropy:{}".format(cross_entropy))
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name + '/loss', loss)
    return loss


# --------------------------------------------------------------------------
# loss损失值优化
# 输入参数：loss。learning_rate，学习速率。
# 返回参数：train_op，训练op，这个参数要输入sess.run中让模型去训练。
def trainning(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


# -----------------------------------------------------------------------
# 评价/准确率计算
# 输入参数：logits，网络计算值。labels，标签，也就是真实值，在这里是0或者1。
# 返回参数：accuracy，当前step的平均准确率，也就是在这些batch中多少张图片被正确分类了。
def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy

# ========================================================================
