"""A strawman model for Kaggle Planet data, includes a pooling layer and a linear layer """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import argparse
from collections import OrderedDict
import numpy as np
from scipy.special import logit
import tensorflow as tf

from utils import KagglePlanetImage as Image, KagglePlanetImageLabels as ImageLabels, DATA_DIR
from read_data import read_data_sets

FLAGS = None

def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)

  # Build the feature dict with keys image_raw, weather, common, special
  feature_dict = {'image_raw': tf.FixedLenFeature([], tf.string)}
  for label_type, labels in ImageLabels.LABELS.items():
    shape = [len(labels)]
    feature_dict[label_type] = tf.FixedLenFeature(shape, tf.int64)

  features = tf.parse_single_example(serialized_example, features=feature_dict)

  # Convert from a scalar string tensor to a uint8 tensor
  image = tf.decode_raw(features['image_raw'], tf.float64)
  image = tf.cast(image, tf.float32)
  image.set_shape([Image.SIZE])

  labels = OrderedDict()
  for label_type  in ImageLabels.LABELS:
    labels[label_type] = tf.cast(features[label_type], tf.float32)

  return image, labels


def debug_inputs(filename):
  " show tfrecords image for debug. "
  filename_queue = tf.train.string_input_producer([filename])
  image, labels = read_and_decode(filename_queue)

  init_op = tf.initialize_all_variables()
  with tf.Session() as sess:
      sess.run(init_op)
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      for i in range(4):
          print(image.get_shape())
          print(image.eval().shape)
          print(labels['weather'].get_shape())
          print(labels['weather'].eval())
          print(labels['common'].get_shape())
          print(labels['common'].eval())

def inputs(filename, batch_size, num_epochs):
  """Reads input data num_epochs times.
  Args:
    filename: input filename to read from
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input data, or 0/None to
       train forever.
  Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, mnist.NUM_CLASSES).
    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
  """
  if not num_epochs: num_epochs = None

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
        [filename], num_epochs=num_epochs)

    # Even when reading in multiple threads, share the filename
    # queue.
    image, image_labels = read_and_decode(filename_queue)

    # Shuffle the examples and collect them into batch_size batches. Runs in two threads to avoid
    # being a bottleneck. While image_labels was a dict, shuffle batch expects tensors. So this will
    # return a list of tensors which we reconstruct into a dict.
    shuffled = tf.train.shuffle_batch(
        [image] + image_labels.values(), batch_size=batch_size, num_threads=1,
        capacity=1000 + 3 * batch_size,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=1000)

    images = shuffled[0]
    labels = OrderedDict()
    for label_type, label_tensor in zip(image_labels.keys(), shuffled[1:]):
      labels[label_type] = label_tensor

    return images, labels


def get_metric_ops(logits, labels, cutoff=0.5):
  """ Build metric ops with uniform cutoff """
  # we have multiple label classes (weather/common/special) but the metrics apply to all labels
  # together. Other types are multilabeled.
  # TODO: Replace with metrics from tf.conrib.metrics 
  def confusion_matrix(preds, labels):
    "Count true positives, true negatives, false positives and false negatives"
    tp = tf.count_nonzero(preds * labels, axis=1)
    fp = tf.count_nonzero(preds * (1 - labels), axis=1)
    tn = tf.count_nonzero((1 - preds) * (1 - labels), axis=1)
    fn = tf.count_nonzero((1 - preds) * labels, axis=1)
    return tp, fp, tn, fn

  # Weather labels are mutually exclusive, so we pick argmax for the prediction and one-hot encode
  # them so we can count tp, fp, etc.
  preds = tf.one_hot(tf.argmax(logits['weather'], axis=1), depth=len(ImageLabels.LABELS['weather']))
  tp, fp, tn, fn = confusion_matrix(preds, labels['weather'])

  # We currently do not predict special labels. Add common label results to the current counts.
  preds = tf.greater(logits['common'], logit(cutoff))
  preds = tf.cast(preds, tf.float32)
  common_tp, common_fp, common_tn, common_fn = confusion_matrix(preds, labels['common'])
  tp += common_tp
  fp += common_fp
  tn += common_tn
  fn += common_fn
      
  # Calculate precision, recall and F2 score.
  precision = tp / (tp + fp)
  recall = tp / (tp + fn)
  f2_score = (5 * precision * recall) / (4 * precision + recall)

  # Replace nans with zeros
  def replace_nans(tensor):
    return tf.where(tf.is_nan(tensor), tf.zeros_like(tensor), tensor)

  precision = replace_nans(precision)
  recall = replace_nans(recall)
  f2_score = replace_nans(f2_score)

  return tf.reduce_mean(precision), tf.reduce_mean(recall), tf.reduce_mean(f2_score)


def run_training(training_filename):
  """Train MNIST for a number of steps."""
  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Input images and labels. Default values for now, read from args later
    images, labels = inputs(training_filename, batch_size=50, num_epochs=1)

    # Build a Graph that computes predictions from the inference model.
    logits = strawman(images)

    # Add to the Graph the loss calculation. Weather labels are mutually exclusive, so we use softmax
    # other labels are not exclusive, so use a sigmoid. For now don't include special labels in the
    # loss function
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                 labels=labels['weather'], logits=logits['weather'])) + \
              tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                 targets=labels['common'], logits=logits['common']))

    train_op = tf.train.AdamOptimizer(1e-4).minimize(loss_op)

    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    precision_op, recall_op, f2_op = get_metric_ops(logits, labels, cutoff=0.1)

    # Create a session for running operations in the Graph.
    sess = tf.Session()

    # Initialize the variables (the trained variables and the
    # epoch counter).
    sess.run(init_op)

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
      step = 0
      while not coord.should_stop():
        start_time = time.time()
        is_summary_step = (step % 25) == 0

        ops_to_compute = [train_op]
        if is_summary_step:
          ops_to_compute.extend([loss_op, f2_op])

        op_values = sess.run(ops_to_compute)
        duration = time.time() - start_time

        # Print an overview fairly often.
        if is_summary_step:
          loss_value, f2_score = op_values[1:]
          # Add metrics to TensorBoard.    
          # tf.summary.scalar('Precision', precision)
          # tf.summary.scalar('Recall', recall)
          # tf.summary.scalar('f2-score', f2_score)

          print('Step {}: loss = {:.3f}, F2 = {:.3f}  ({:.3f} sec)'
                .format(step, loss_value, f2_score, duration))

        step += 1
    except tf.errors.OutOfRangeError:
      print('Done training for %d epochs, %d steps.' % (1, step))
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()



def strawman(x):
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

  # Map the 1024 features to weather and common classes
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



# def conv2d(x, W):
#   """conv2d returns a 2d convolution layer with full stride."""
#   return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


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

def get_filename(dataset, start_idx, end_idx):
  """ Get filename with given start, end indices. No filename validation. """
  assert dataset in ['train', 'test', 'validation'], \
         "dataset must be one of train, test, validation"
  return os.path.join(DATA_DIR, 'protobuf', 
                      'images.{}.{}_{}.proto'.format(dataset, start_idx, end_idx))

def main(_):
  filename = get_filename('train', 0, 1999)
  # debug_inputs(filename)
  run_training(filename)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
