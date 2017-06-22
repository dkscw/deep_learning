"""Read input images and labels """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import tensorflow as tf

from planet_utils import KagglePlanetImage as Image, KagglePlanetImageLabels as ImageLabels


def read_and_decode(filename_queue):
  """ Read and decode images and labels from protobuf format """
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)

  # Build the feature dict with keys image_raw, weather, common, special
  feature_dict = {'image_raw': tf.FixedLenFeature([], tf.string)}
  for label_type, labels in ImageLabels.LABELS.items():
    shape = [len(labels)]
    feature_dict[label_type] = tf.FixedLenFeature(shape, tf.int64)

  features = tf.parse_single_example(serialized_example, features=feature_dict)

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

  init_op = tf.group(tf.global_variables_initializer(),
                     tf.local_variables_initializer())
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


def inputs(filename, batch_size, num_epochs, shuffle=True):
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
  if not num_epochs:
    num_epochs = None

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)

    # Even when reading in multiple threads, share the filename queue.
    image, image_labels = read_and_decode(filename_queue)

    # Shuffle the examples and collect them into batch_size batches. Runs in two threads to avoid
    # being a bottleneck. While image_labels was a dict, shuffle batch expects tensors. So this will
    # return a list of tensors which we reconstruct into a dict.
    if shuffle:
      shuffled = tf.train.shuffle_batch(
          [image] + image_labels.values(), batch_size=batch_size, num_threads=1,
          capacity=1000 + 3 * batch_size, min_after_dequeue=1000)

      images = shuffled[0]
      labels = OrderedDict()
      for label_type, label_tensor in zip(image_labels.keys(), shuffled[1:]):
        labels[label_type] = label_tensor

    return images, labels

