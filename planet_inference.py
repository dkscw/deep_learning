""" Inference and loss ops """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import tensorflow as tf

from planet_utils import KagglePlanetImage as Image, KagglePlanetImageLabels as ImageLabels


def inference(x):
  """builds the graph for a strawman for classifying digits.

  Args:
    x: an input tensor with the dimensions (N_examples, Image.SIZE)

  Returns:
    y: a tensor of shape (N_examples, N_LABELS), with values
    equal to the logits of classifying the digit into the label classes.
  """
  # Reshape to use within a convolutional neural net.
  x_image = tf.reshape(x, [-1, Image.HEIGHT, Image.WIDTH, Image.DEPTH])

  # Pooling layer - downsamples by 16X to dimension 16x16x4
  filter_size = 16
  h_pool1 = max_pool(x_image, filter_size)

  # Flatten
  flat_dim = int(Image.SIZE / (filter_size) ** 2)
  h_pool1_flat = tf.reshape(h_pool1, [-1, flat_dim])

  # Map the features to weather and common classes
  weights = OrderedDict()
  biases = OrderedDict()
  logits = OrderedDict()
  for label_type in ImageLabels.LABELS:
      n_classes = len(ImageLabels.LABELS[label_type])
      weights[label_type] = weight_variable([flat_dim, n_classes])
      biases[label_type] = bias_variable([n_classes])
      logits[label_type] = tf.matmul(h_pool1_flat, weights[label_type]) + biases[label_type]

  return logits

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  # W_conv1 = weight_variable([5, 5, 1, 32])
  # b_conv1 = bias_variable([32])
  # h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  # W_fc1 = weight_variable([7 * 7 * 64, 1024])
  # b_fc1 = bias_variable([1024])

  # h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  # h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features. This is a placeholder feature that for now we just set at 1.
  # keep_prob = tf.placeholder(tf.float32)
  # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)



def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x, filter_size):
  """downsamples a feature map by filter_size"""
  return tf.nn.max_pool(x, ksize=[1, filter_size, filter_size, 1],
                        strides=[1, filter_size, filter_size, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def loss(logits, labels):
  """ Build the loss op given logits, labels. Both are dicts with keys 'weather', 'common'.
  Weather labels are mutually exclusive so we use cross entropy with softmax. Ground labels are
  multi-class, so we use sigmoid cross entropy.

  Note: the softmax x-entropy returns a 1-D tensor, whereas sigmoid x-entropy returns a 2-D tensor
  with the same shape as logits. Investigate whether summing the means is the right thing to do. """

  return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                        labels=labels['weather'], logits=logits['weather'])) + \
         tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        targets=labels['common'], logits=logits['common']))
