import os
import math
import numpy as np
import tensorflow as tf
import input_data_new as input_data  # 更改为新的数据预处理文件名称
import model_resnet18 as model  # 更改为新的模型文件名称
import time
import requests
import json
from urllib.request import urlretrieve
from sklearn.model_selection import train_test_split
from PIL import Image
from io import BytesIO


#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#N_CLASSES = 2
#IMG_W = 224
#IMG_H = 224
#BATCH_SIZE = 2
#CAPACITY = 256
#learning_rate = 0.0001

#response = requests.get(api_endpoint)
#data = json.loads(response.text)

#os.makedirs("images", exist_ok=True)
#os.makedirs("labels", exist_ok=True)

# 前30张图片为合格品
#for i, item in enumerate(data[:30]):
#  if not isinstance(item, dict):
#    print(f"Unexpected data type at index {i}: {item}")
#    continue

#    image_url = item["image_url"]
#    img_filename = f"images/qualified_{i + 1}.jpg"
#    urlretrieve(image_url, img_filename)

#    label_filename = f"labels/qualified_{i + 1}.txt"
#    with open(label_filename, "w") as f:
#        f.write("1")  # 合格品标签为1#

# 后30张图片为不合格品
#for i, item in enumerate(data[-30:]):
#  if not isinstance(item, dict):
#    print(f"Unexpected data type at index {i}: {item}")
#    continue
#
#    image_url = item["image_url"]
#   img_filename = f"images/unqualified_{i + 1}.jpg"
#    urlretrieve(image_url, img_filename)
#    label_filename = f"labels/unqualified_{i + 1}.txt"
#    with open(label_filename, "w") as f:
#        f.write("0")  # 不合格品标签为0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

N_CLASSES = 2
IMG_W = 224
IMG_H = 224
BATCH_SIZE = 2
CAPACITY = 256
learning_rate = 0.0001


def get_image_records(task_id, offset=0, limit=60):
    resp = requests.get(
        url='http://backend-server-container:9090/api/image',
        params=dict(
            task_category_id=task_id,
            offset=offset,
            limit=limit
        )
    )

    if resp.status_code == 200:
        return resp.json()
    else:
        raise RuntimeError(resp.text)


def get_image_by_id(image_id):
    url = f"http://backend-server-container:9090/api/image/{image_id}"
    r = requests.get(url)

    if r.status_code == 200:
        return Image.open(io.BytesIO(r.content))
    else:
        raise RuntimeError(r.text)


def save_label(label, label_filename):
    with open(label_filename, "w") as f:
        f.write(str(label))

task_id = "your_task_id_here"  # 请替换为实际的任务ID
image_records = get_image_records(task_id)

data = image_records[:30] + image_records[30:]  # 前30张合格，后30张不合格

image_folder = "images"
label_folder = "labels"

if not os.path.exists(image_folder):
    os.makedirs(image_folder)

if not os.path.exists(label_folder):
    os.makedirs(label_folder)

# 保存图像和标签
for i, item in enumerate(data[:30]):
    image_id = item["id"]
    img = get_image_by_id(image_id)
    img_filename = os.path.join(image_folder, f"qualified_{i + 1}.jpg")
    img.save(img_filename)

    label_filename = os.path.join(label_folder, f"qualified_{i + 1}.txt")
    save_label("1", label_filename)  # 合格品标签为1

for i, item in enumerate(data[-30:]):
    image_id = item["id"]
    img = get_image_by_id(image_id)
    img_filename = os.path.join(image_folder, f"unqualified_{i + 1}.jpg")
    img.save(img_filename)

    label_filename = os.path.join(label_folder, f"unqualified_{i + 1}.txt")
    save_label("0", label_filename)  # 不合格品标签为0

image_filenames = []
labels = []

for label_filename in os.listdir(label_folder):
    with open(os.path.join(label_folder, label_filename), "r") as f:
        label = int(f.read().strip())
        image_filename = os.path.join(image_folder, f"{label_filename[:-4]}.jpg")

    image_filenames.append(image_filename)
    labels.append(label)

X_train, X_test, y_train, y_test = train_test_split(
    image_filenames, labels, test_size=0.2, random_state=42, stratify=labels
)

print("Training set size:", len(X_train))
print("Test set size:", len(X_test))

train, train_label = input_data.get_files(X_train, y_train)
data_len = len(train)

val, val_label = input_data.get_files(X_test, y_test)
test_each_epoch_step = len(val) // BATCH_SIZE

logs_train_dir = './train_logs/logs/'
logs_checkpoint = './train_logs/checkpoint/'

epochs = 1000
each_epoch_step = math.ceil(data_len / BATCH_SIZE)
MAX_STEP = epochs * each_epoch_step

train_batch, train_label_batch, train_iterator = input_data.train_build(epochs, BATCH_SIZE, train, train_label)
val_batch, val_label_batch, test_iterator = input_data.test_build(BATCH_SIZE, val, val_label)

input_batch_image = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
input_batch_label = tf.placeholder(tf.int64, shape=[BATCH_SIZE, ])
drop_out = tf.placeholder(tf.float32, shape=())

train_logits = model.inference(input_batch_image, BATCH_SIZE, N_CLASSES, drop_out)
train_loss = model.losses(train_logits, input_batch_label)

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(learning_rate, global_step, each_epoch_step, 0.96, staircase=True)
train_op = model.trainning(train_loss, learning_rate, global_step)
train_acc = model.evaluation(train_logits, input_batch_label)

pre_test_acc = 0
sess = tf.Session()
train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)

saver = tf.train.Saver()
sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
sess.run([train_iterator.initializer, test_iterator.initializer])

epoch_index = 0
summary_op = tf.summary.merge_all()
start = time.perf_counter()

if tf.train.latest_checkpoint(logs_checkpoint) is not None:
    print("----------------本次不导入预训练模型-----------------")

for step in range(MAX_STEP):
    tr_batch, tr_bat_label = sess.run([train_batch, train_label_batch])
    _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc],
                                    feed_dict={input_batch_image: tr_batch, input_batch_label: tr_bat_label,
                                               drop_out: 0.4})

    if step % each_epoch_step == 0:
        if step >= each_epoch_step:
            epoch_index = epoch_index + 1
        all_test_acc = 0
        all_test_loss = 0
        for batch in range(test_each_epoch_step):
            va_batch, va_bat_label = sess.run([val_batch, val_label_batch])
            te_logits, te_loss, te_acc = sess.run([train_logits, train_loss, train_acc],
                                                  feed_dict={input_batch_image: va_batch,
                                                             input_batch_label: va_bat_label, drop_out: 1.0,
                                                             })
            all_test_acc = all_test_acc + te_acc
            all_test_loss = all_test_loss + te_loss
        avg_test_acc = all_test_acc / test_each_epoch_step
        avg_test_loss = all_test_loss / test_each_epoch_step
        print('Step %d  %d/epoch:%d------  train loss = %.2f  ------ train accuracy = %.2f%%  ------'
              '    test_loss = %.2f '
              ' ------  test accuracy = %.2f%%' % (step,
                                                   epoch_index,
                                                   epochs, tra_loss,
                                                   tra_acc * 100.0, avg_test_loss, avg_test_acc * 100.0))

        summary_str = sess.run(summary_op, feed_dict={input_batch_image: tr_batch, input_batch_label: tr_bat_label,
                                                      drop_out: 1.0})
        train_writer.add_summary(summary_str, step)
        if avg_test_acc > pre_test_acc:
            checkpoint_path = os.path.join(logs_checkpoint, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)
            print('Save model.ckpt at step %d' % step)
            pre_test_acc = avg_test_acc

end = time.perf_counter()
print('Training completed and the total time is %0.2f seconds' % (end - start))
