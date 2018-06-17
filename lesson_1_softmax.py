from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.INFO)

# Load training and eval data
mnist = tf.contrib.learn.datasets.load_dataset("mnist")

#
x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
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

# stochastic gradient descent（SGD）
# epoch是指把所有训练数据完整的过多少遍，epoch太大缺点有两个…一个是过拟合（overfit）另一个是训练时间太长
# batch是指你一次性拿多少个数据去训练

n = len(images)
epoch = 30
batch = 10
iter_count = int(n / batch)

print("images.len:", n, "epoch:", epoch, "batch:", batch)
seq = np.arange(n)

for e in range(epoch):  
  np.random.shuffle(seq)
  for i in range(iter_count):
    u = i * batch
    s = seq[u : u + batch]
    batch_xs = images[s]
    batch_ys = labels[s]
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
  accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  accuracy = sess.run(accuracy_op, feed_dict={x: eval_data, y_: eval_labels})
  print("train iter:", e, "/", epoch, "accuracy:", accuracy)

def evalImage(index):
  ed = np.reshape(eval_data[index], [1,784])
  ey = tf.nn.softmax(tf.matmul(ed, W) + b)
  ei = tf.argmax(ey, 1)
  rl = sess.run(ei)[0]
  rr = np.argmax(eval_labels[index])
  return rl == rr


def showImage(image):
	plt.title('img')
	plt.imshow(image.reshape((28,28)), cmap = plt.cm.gray_r)
	plt.show()
