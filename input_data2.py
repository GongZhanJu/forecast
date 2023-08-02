import numpy as np
import os
import tensorflow as tf


label_dict = {'1': '0', '2': '1','3':'2','4':'3','5':'4','6':'5','7':'6'}


def get_files(file_dir):
    image_list, label_list = [], []
    for label in os.listdir(file_dir):
        img_dir = file_dir + label
        for img in os.listdir(img_dir):
            img_path = img_dir + '/' + img
            image_list.append(img_path)
            label_list.append(int(label_dict[label]))
    # print('There are %d data' %(len(image_list)))
    temp = np.array([image_list, label_list])
    # print(temp)
    temp = temp.transpose()
    np.random.shuffle(temp)
    np.random.shuffle(temp)
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    return image_list, label_list

#训练集图像预处理
def decode_train(image, label):
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)
    image_string = tf.read_file(image)      # 读取原始文件
    image_decoded = tf.image.decode_jpeg(image_string,channels=3)    # 解码JPEG图片
    # image_resized = tf.image.resize(image_decoded,[150,150]) / 255.0  # 归一化
    image_resized = tf.image.resize_images(image_decoded, (224, 224))
    image_resized = image_resized/255.0
    return image_resized,label

#测试集图像预处理
def decode_test(image, label):
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)
    image_string = tf.read_file(image)      # 读取原始文件
    image_decoded = tf.image.decode_jpeg(image_string,channels=3)    # 解码JPEG图片
    # image_resized = tf.image.resize(image_decoded,[150,150]) / 255.0  # 归一化
    image_resized = tf.image.resize_images(image_decoded, (224, 224))
    image_resized = image_resized/255.0
    return image_resized,label


# 构建训练数据集
def train_build(epochs,batch_size,train_filenames,train_labels):

    train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames,train_labels))
    # map多进程执行
    train_dataset = train_dataset.map(
        map_func = decode_train,
        num_parallel_calls = 1)
    # 取出前buffer_size个数据放入buffer，并从中随机采样，采样后的数据用后续数据替换
    train_dataset = train_dataset.shuffle(buffer_size=500)     # 缓冲区
    train_dataset = train_dataset.repeat()                # 无限重复
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(500)    # 优化
    iterator = train_dataset.make_initializable_iterator()
    img, label = iterator.get_next()
    return img, label,iterator


# 构建测试数据集
def test_build(batch_size,train_filenames,train_labels):

    test_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
    test_dataset = test_dataset.map(decode_test)
    test_dataset = test_dataset.repeat()
    test_dataset = test_dataset.batch(batch_size)
    iterator = test_dataset.make_initializable_iterator()
    img, label = iterator.get_next()
    return img, label, iterator


# if __name__ == '__main__':
#     file_dir = '../data/train/'  # 训练样本的读入路径
#     data, label = get_files(file_dir)
#     # print(data)
#     # print()
#     # print(label)
# #     # dataset = tf.data.Dataset.from_tensor_slices((data, label))
# #     # dataset = dataset.map(_decode_and_resize)
# #     # dataset = dataset.batch(5)  # 改为1，方便计数
# #     # dataset = dataset.repeat(20000000000000000)  # 数据集重复两次
# #     # # print(dataset)
# #     # iterator = dataset.make_one_shot_iterator()
# #     # img, label = iterator.get_next()
# #     img_batch, label_batch,iterator = test_build(5,data,label)
# #     input_batch_image = tf.placeholder(tf.float32, shape=[32, 150, 150, 3])  # 输入数据batch
# #     input_batch_label = tf.placeholder(tf.int32, shape=[32, ])
#     train_batch, train_label_batch, train_iterator = train_build(1000, 32, data, label)
#     print(train_batch)
#     print(train_label_batch)
# #
#     with tf.Session() as sess:
#         sess.run(train_iterator.initializer)
#         # while 1:
#         #     image, label = sess.run([train_batch, train_label_batch])
#             # print(label)

