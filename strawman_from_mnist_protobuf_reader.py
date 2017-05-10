"""A strawman model for Kaggle Planet data, includes a pooling layer and a linear layer """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import argparse
import tensorflow as tf

from utils import KagglePlanetImage as Image, KagglePlanetImageLabels as ImageLabels, DATA_DIR
from read_data import read_data_sets

FLAGS = None

N_LABELS = len(ImageLabels.COMMON_LABELS)

def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'height': tf.FixedLenFeature([], tf.int64),
          'width': tf.FixedLenFeature([], tf.int64),
          'depth': tf.FixedLenFeature([], tf.int64),
          'image_raw': tf.FixedLenFeature([], tf.string),
          'labels': tf.FixedLenFeature([], tf.string)
      })

  # Convert from a scalar string tensor (whose single string has
  # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
  # [mnist.IMAGE_PIXELS].
  image = tf.decode_raw(features['image_raw'], tf.float64)
  image = tf.cast(image, tf.float32)
  image.set_shape([Image.SIZE])

  # # OPTIONAL: Could reshape into a 28x28 image and apply distortions
  # # here.  Since we are not applying any distortions in this
  # # example, and the next step expects the image to be flattened
  # # into a vector, we don't bother.

  # # Convert from [0, 255] -> [-0.5, 0.5] floats.
  # image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

  # Labels are saved as a string of length 10, but for some reason reading as tf.uint8
  # blows up by a factor of 8. Should probably save them as ints. 
  labels = tf.decode_raw(features['labels'], tf.int64)
  labels = tf.cast(labels, tf.float32)
  labels.set_shape([N_LABELS])
  return image, labels


def debug_inputs():
  '''
  show tfrecords image for debug.
  '''
  filename = os.path.join(DATA_DIR, 'protobuf', 'train.0_10000.tfrecords')
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
          print(labels.get_shape())
          print(labels.eval())

def inputs(batch_size, num_epochs):
  """Reads input data num_epochs times.
  Args:
    train: Selects between the training (True) and validation (False) data.
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
  filename = os.path.join(DATA_DIR, 'protobuf', 'train.0_10000.tfrecords')

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
        [filename], num_epochs=num_epochs)

    # Even when reading in multiple threads, share the filename
    # queue.
    image, labels = read_and_decode(filename_queue)

    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    images, labels = tf.train.shuffle_batch(
        [image, labels], batch_size=batch_size, num_threads=1,
        capacity=1000 + 3 * batch_size,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=1000)
    return images, labels

def run_training():
  """Train MNIST for a number of steps."""

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Input images and labels. Default values for now, read from args later
    images, labels = inputs(batch_size=50, num_epochs=1)

    # Build a Graph that computes predictions from the inference model.
    logits = strawman(images)

    # Add to the Graph the loss calculation.
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(targets=labels, logits=logits))

    train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)

    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

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

        # Run one step of the model.  The return values are
        # the activations from the `train_op` (which is
        # discarded) and the `loss` op.  To inspect the values
        # of your ops or variables, you may include them in
        # the list passed to sess.run() and the value tensors
        # will be returned in the tuple from the call.
        _, loss_value = sess.run([train_op, loss])

        duration = time.time() - start_time

        # Print an overview fairly often.
        if step % 25 == 0:
          print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value,
                                                     duration))
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

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  # W_conv1 = weight_variable([5, 5, 1, 32])
  # b_conv1 = bias_variable([32])
  # h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 16X to dimension 16x16x4
  filter_size = 16
  h_pool1 = max_pool(x_image, filter_size)

  # Flatten
  flat_dim = int(Image.SIZE / (filter_size) ** 2)
  h_pool1_flat = tf.reshape(h_pool1, [-1, flat_dim])

  # Second convolutional layer -- maps 32 feature maps to 64.
  # W_conv2 = weight_variable([5, 5, 32, 64])
  # b_conv2 = bias_variable([64])
  # h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  # h_pool2 = max_pool_2x2(h_conv2)

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

  # Map the 1024 features to 10 classes, one for each digit
  W_fc2 = weight_variable([flat_dim, N_LABELS])
  b_fc2 = bias_variable([N_LABELS])

  y_conv = tf.matmul(h_pool1_flat, W_fc2) + b_fc2
  return y_conv  # , keep_prob


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


def main(_):
  # debug_inputs()
  run_training()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # parser.add_argument('--data_dir', type=str,
  #                     default='/tmp/tensorflow/mnist/input_data',
  #                     help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
