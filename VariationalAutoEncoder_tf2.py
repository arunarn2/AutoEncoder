import os
import random
import shutil

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.misc import imresize
from scipy.misc import imsave
from tensorflow.examples.tutorials.mnist import input_data


class Plots:
    def __init__(self, dir, n_img_x=8, n_img_y=8, img_w=28, img_h=28, resize_factor=1.0):
        self.dir = dir
        assert n_img_x > 0 and n_img_y > 0
        assert img_w > 0 and img_h > 0
        assert resize_factor > 0

        self.n_img_x = n_img_x
        self.n_img_y = n_img_y
        self.n_tot_imgs = n_img_x * n_img_y
        self.img_w = img_w
        self.img_h = img_h
        self.resize_factor = resize_factor

    def save_images(self, images, name='result.jpg'):
        images = images.reshape(self.n_img_x*self.n_img_y, self.img_h, self.img_w)
        imsave(self.dir + "/"+name, self.merge(images, [self.n_img_y, self.n_img_x]))

    def merge(self, images, size):
        h, w = images.shape[1], images.shape[2]
        h_ = int(h * self.resize_factor)
        w_ = int(w * self.resize_factor)
        img = np.zeros((h_ * size[0], w_ * size[1]))
        for idx, image in enumerate(images):
            i = int(idx % size[1])
            j = int(idx / size[1])
            image_ = imresize(image, size=(w_, h_), interp='bicubic')
            img[j*h_:j*h_+h_, i*w_:i*w_+w_] = image_
        return img


# borrowed from https://gist.github.com/jakevdp/91077b0cae40f8f8244a
def discrete_cmap(n, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    base = plt.cm.get_cmap(base_cmap)
    return base.from_list(base.name + str(n), base(np.linspace(0, 1, n)), n)


mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
test = np.column_stack((mnist.test.images, mnist.test.labels))

test_pairs = []
for i in range(10):
    test_pairs.append(test[test[:, 784] == i])

same_digit_pairs = []
for i in range(10):
    same_digit_pairs.append(random.sample(test_pairs[i], 2))
same_digit_pairs = np.array(same_digit_pairs)
same_digit_pairs = same_digit_pairs.reshape(20, 785)
x_same = same_digit_pairs[:, :784]
y_s = same_digit_pairs[:, 784]

diff_digit_pairs = []
random.seed(345)
for i in range(10):
    a = random.sample(test, 2)
    while a[0][784] == a[1][784]:
        a = random.sample(test, 2)
    diff_digit_pairs.append(a)

diff_digit_pairs = np.array(diff_digit_pairs)
diff_digit_pairs = diff_digit_pairs.reshape(20, 785)
x_diff = diff_digit_pairs[:, :784]
y_d = diff_digit_pairs[:, 784]

# Reformatting labels with 16 one-hot encoding
y_same = np.zeros((y_s.shape[0], 10), dtype="int64")
y_diff = np.zeros((y_d.shape[0], 10), dtype="int64")

for i in xrange(y_s.shape[0]):
    y_same[i, int(y_s[i])] = 1
for i in xrange(y_d.shape[0]):
    y_diff[i, int(y_d[i])] = 1

del y_s, y_d, diff_digit_pairs, same_digit_pairs, test_pairs, mnist, test

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
n_samples = mnist.train.num_examples
learning_rate = 0.001
batch_size = 128
training_epochs = 5
display_step = 5
n_hidden = 512
n_input = n_output = 784
n_z = 2
total_batch = int(n_samples / batch_size)
min_tot_loss = 1e99

x = tf.placeholder(tf.float32, [None, 784], name='n_input')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# encoder
with tf.variable_scope("encoder"):
    # initializers
    w_init = tf.contrib.layers.variance_scaling_initializer()
    b_init = tf.constant_initializer(0.)

    # 1st hidden layer - ELU takes care of vanishing gradient problem
    w0 = tf.get_variable('w0', [x.get_shape()[1], n_hidden], initializer=w_init)
    b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
    h0 = tf.nn.dropout(tf.nn.elu(tf.matmul(x, w0) + b0), keep_prob)

    # 2nd hidden layer
    w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
    b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
    h1 = tf.nn.dropout(tf.nn.tanh(tf.matmul(h0, w1) + b1), keep_prob)

    # output layer
    wo = tf.get_variable('wo', [h1.get_shape()[1], n_z * 2], initializer=w_init)
    bo = tf.get_variable('bo', [n_z * 2], initializer=b_init)
    parameters = tf.matmul(h1, wo) + bo

    # The mean parameter is unconstrained
    mu = parameters[:, :n_z]
    # The standard deviation must be positive. Parametrize with a softplus and
    # add a small epsilon for numerical stability
    sigma = 1e-6 + tf.nn.softplus(parameters[:, n_z:])

with tf.variable_scope("latent_var"):
    z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
    # initializers
    w_init = tf.contrib.layers.variance_scaling_initializer()
    b_init = tf.constant_initializer(0.)

    # 1st hidden layer
    w0 = tf.get_variable('w0', [z.get_shape()[1], n_hidden], initializer=w_init)
    b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
    h0 = tf.nn.dropout(tf.nn.tanh(tf.matmul(z, w0) + b0), keep_prob)

    # 2nd hidden layer
    w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
    b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
    h1 = tf.nn.dropout(tf.nn.elu(tf.matmul(h0, w1) + b1), keep_prob)

    # output layer-mean
    wo = tf.get_variable('wo', [h1.get_shape()[1], n_output], initializer=w_init)
    bo = tf.get_variable('bo', [n_output], initializer=b_init)
    y = tf.sigmoid(tf.matmul(h1, wo) + bo)

# loss
marginal_likelihood = tf.reduce_mean(tf.reduce_sum(x * tf.log(1e-10 + y) + (1 - x) * tf.log(1e-10 + 1 - y), 1))
KL_divergence = tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) -
                                                   tf.log(1e-10 + tf.square(sigma)) - 1, 1))
neg_marginal_likelihood = -marginal_likelihood
loss = -(marginal_likelihood - KL_divergence)

# Use ADAM optimizer
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

nimages = 10
test_images = mnist.test.images[0:nimages*nimages, :].reshape(nimages*nimages, 28, 28)
dirpath = os.path.join(os.getcwd(), "output")
if os.path.exists(dirpath):
    shutil.rmtree(dirpath)
os.mkdir("output")
myplot = Plots(dirpath, nimages, nimages, 28, 28, 1.0)
myplot.save_images(test_images, name='input.jpg')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer(), feed_dict={keep_prob: 0.9})

    for epoch in range(training_epochs):
        for i in range(total_batch):
            batch_xs, _ = mnist.train.next_batch(batch_size)
            _, tot_loss, loss_likelihood, loss_div = sess.run((train, loss, neg_marginal_likelihood, KL_divergence),
                                                              feed_dict={x: batch_xs, keep_prob: 0.9})

        print("epoch %d: Total Loss: %03.2f; Loss likelihood %03.2f; Loss divergence %03.2f" %
              (epoch, tot_loss, loss_likelihood, loss_div))

        if min_tot_loss > tot_loss or epoch + 1 == training_epochs:
            min_tot_loss = tot_loss

            test_images = mnist.test.images[0:nimages * nimages, :]
            y_img = sess.run(y, feed_dict={x: test_images, keep_prob: 1})
            y_img_img = y_img.reshape(nimages * nimages, 28, 28)
            myplot.save_images(y_img_img, name="/epoch_%02d" % epoch + ".jpg")

    nx = ny = 20
    x_values = np.linspace(-2, 2, nx)
    y_values = np.linspace(-2, 2, ny)

    canvas = np.empty((28 * ny, 28 * nx))
    for i, yi in enumerate(x_values):
        for j, xi in enumerate(y_values):
            z_mu = np.array([[xi, yi]] * 100)
            x_mean = sess.run(y, feed_dict={z: z_mu, keep_prob: 1.0})
            canvas[(nx - i - 1) * 28:(nx - i) * 28, j * 28:(j + 1) * 28] = x_mean[0].reshape(28, 28)

    plt.figure(figsize=(10, 10))
    Xi, Yi = np.meshgrid(x_values, y_values)
    plt.imshow(canvas, origin="upper", cmap="gray")
    plt.tight_layout()
    plt.show()

    x_sample, y_sample = mnist.test.next_batch(5000)
    z_mu = sess.run(z, feed_dict={x: x_sample, keep_prob: 1.0})
    plt.figure(figsize=(10, 10))
    plt.scatter(z_mu[:, 0], z_mu[:, 1], cmap=discrete_cmap(10, 'jet'), c=np.argmax(y_sample, 1), edgecolors='black')
    plt.colorbar()
    plt.grid()
    plt.show()

    # code_same_1 = mu.eval(mu, feed_dict={x: x_same[0], keep_prob: 1.0})
    # code_same_2 = mu.eval(mu, feed_dict={x: x_same[1], keep_prob: 1.0})
    # y_same = y.eval(y, feed_dict={})

    # z_same = sess.run(z, feed_dict={x: x_same, keep_prob: 1.0})
    # plt.figure(figsize=(10, 10))
    # plt.scatter(z_same[:, 0], z_same[:, 1], cmap='CMRmap', c=np.argmax(y_same, 1))
    # plt.colorbar()
    # plt.grid()
    # plt.show()
    #
    # z_diff = sess.run(z, feed_dict={x: x_same, keep_prob: 1.0})
    # plt.figure(figsize=(10, 10))
    # plt.scatter(z_mu[:, 0], z_mu[:, 1], cmap='RdYlGn_r', c=np.argmax(y_sample, 1))
    # plt.colorbar()
    # plt.grid()
    # plt.show()




