from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def showImg(image, shape):
	plt.title('img')
	plt.imshow(image.reshape(shape), cmap = plt.cm.gray_r)
	plt.show()

mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images  # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_data = mnist.test.images  # Returns np.array
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

#image: [batch, in_height, in_width, in_channels]
x_img = train_data[0]
x_img = np.asarray(x_img, dtype=np.float32)
x_img = tf.reshape(x_img, [1, 28, 28, 1])
input = tf.Variable(x_img)

#Filter: [filter_height, filter_width, in_channels, out_channels]
sobel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype = np.float32)
kernel = tf.Variable(sobel)
kernel = tf.reshape(kernel, [3, 3, 1, 1])
filter = tf.Variable(kernel)
op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

feature = sess.run(op)
fea_img = np.asarray(feature, dtype=np.float32)
showImg(fea_img, [28, 28])

