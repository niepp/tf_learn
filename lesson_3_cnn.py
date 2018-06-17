from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.INFO)

session_config = tf.ConfigProto()
session_config.gpu_options.allow_growth = True

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# Load training and eval data
mnist = tf.contrib.learn.datasets.load_dataset("mnist")

x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None, 10])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy_op = tf.reduce_sum(tf.cast(correct_prediction, "float"))

init = tf.global_variables_initializer()
sess = tf.Session(config = session_config)
sess.run(init)

# train data
images = mnist.train.images
labels = np.asarray(mnist.train.labels, dtype=np.float32)
labels = tf.one_hot(labels, 10, dtype=np.float32)
labels = sess.run(labels)

# evaluate data
eval_data = mnist.test.images  # Returns np.array
eval_labels = np.asarray(mnist.test.labels, dtype=np.float32)
eval_labels = tf.one_hot(eval_labels, 10, dtype=np.float32)
eval_labels = sess.run(eval_labels)

n = len(images)
epoch = 10
batch = 50
iter_count = int(n / batch)

print("images.len:", n, "epoch:", epoch, "batch:", batch)
seq = np.arange(n)

eval_len = len(eval_data)
eval_seq = np.arange(eval_len)
eval_batch = 1000
eval_iter = int(np.ceil(eval_len / eval_batch))

for e in range(epoch):
  np.random.shuffle(seq)
  for i in range(iter_count):
    u = i * batch
    s = seq[u : u + batch]
    batch_xs = images[s]
    batch_ys = labels[s]
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
  correct = 0
  sum = 0  
  for i in range(eval_iter):
    u = i * eval_batch
    s = eval_seq[u : u + eval_batch]
    sum += len(s)
    eval_x = eval_data[s]
    eval_y = eval_labels[s]
    correct += sess.run(accuracy_op, feed_dict={x: eval_x, y_: eval_y, keep_prob: 1.0})
  print("train iter:", e, "/", epoch, " accuracy:", correct / sum)