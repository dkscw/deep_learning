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
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'planet_train')


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
  if not num_epochs: num_epochs = None

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


def get_prediction_ops(logits, labels, cutoff):
  """ Return the weather and ground predictions. """
  weather_preds = tf.argmax(logits['weather'], axis=1)
  ground_preds = tf.greater(logits['common'], logit(cutoff))
  return weather_preds, tf.cast(ground_preds, tf.float32)

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
  weather_preds, ground_preds = get_prediction_ops(logits, labels, cutoff)
  
  # One-hot encode weather preds to count tp, fp, etc. Accuracy for weather prediction is just % of TP
  one_hot_weather_preds = tf.one_hot(weather_preds, depth=len(ImageLabels.LABELS['weather']))
  weather_tp, weather_fp, weather_tn, weather_fn = confusion_matrix(one_hot_weather_preds, labels['weather'])
  weather_accuracy = tf.reduce_mean(tf.cast(weather_tp, tf.float32))

  # We currently do not predict special labels. Add common label results to the current counts.
  common_tp, common_fp, common_tn, common_fn = confusion_matrix(ground_preds, labels['common'])
  tp = weather_tp + common_tp
  fp = weather_fp + common_fp
  tn = weather_tn + common_tn
  fn = weather_fn + common_fn
      
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

  metrics = OrderedDict([
    ('precision', tf.reduce_mean(precision)),
    ('recall', tf.reduce_mean(recall)),
    ('f2_score', tf.reduce_mean(f2_score)),
    ('weather_accuracy', weather_accuracy),
  ])
  
  return metrics


def run_training(filename, params):
  """Train MNIST for a number of steps."""
  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Input images and labels. Default values for now, read from args later
    images, labels = inputs(filename, batch_size=100, num_epochs=1)

    # Build a Graph that computes predictions from the inference model.
    logits = inference(images)
    loss_op = loss(logits=logits, labels=labels)
    train_op = tf.train.AdamOptimizer(1e-4).minimize(loss_op)

    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    metric_ops = get_metric_ops(logits, labels, cutoff=params['cutoff'])

    # Create a session and a saver
    sess = tf.Session()
    sess.run(init_op)
    saver = tf.train.Saver(tf.global_variables())

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    def summary_str(op_values):
      return ', '.join(['{} = {:.3f}'.format(key, val) for key, val in op_values.items()])

    try:
      step = 0
      while not coord.should_stop():
        start_time = time.time()
        is_summary_step = (step % 25) == 0

        ops_to_compute = OrderedDict([
          ('train', train_op),
          ('loss', loss_op)
        ])

        if is_summary_step:
          ops_to_compute.update([
            ('f2', metric_ops['f2_score']),
            ('weather_accuracy', metric_ops['weather_accuracy']),
          ])

        op_values = OrderedDict(zip(ops_to_compute.keys(), sess.run(ops_to_compute.values())))
        # Discard the training op
        op_values.pop('train')

        duration = time.time() - start_time

        # Print an overview fairly often.
        if is_summary_step:
          # Add metrics to TensorBoard.    
          # tf.summary.scalar('Precision', precision)
          # tf.summary.scalar('Recall', recall)
          # tf.summary.scalar('f2-score', f2_score)
          print('Step {}: {} ({:.3f} sec)'.format(step, summary_str(op_values), duration))

        step += 1
    except tf.errors.OutOfRangeError:
      print('Done training for %d epochs, %d steps.' % (1, step))
      # TODO: Report
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()

    # Save a checkpoint
    saver.save(sess, os.path.join(CHECKPOINT_DIR, 'model.ckpt'), global_step=step)
    print("Saved checkpoint")

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()


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


def get_global_step_from_checkpoint(ckpt):
    return ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

def restore_checkpoint(saver, sess):
    ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
    if not ckpt or not ckpt.model_checkpoint_path: 
      raise IOError('No checkpoint file found')

    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
    # Extract global_step from model_checkpoint_path, assuming it looks something like:
    # /project_path/train/model.ckpt-0
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

      return ckpt, global_step


def get_labels_from_predictions(weather_preds, ground_preds):
  """ Get the prediction labels. Weather predictions are the labels index; ground preds are a binary
  matrix """
  labels = []
  for idx in range(weather_preds.shape[0]):
    weather_label = ImageLabels.LABELS['weather'][weather_preds[idx]]
    ground_labels = [ImageLabels.LABELS['common'][col] for col in np.nonzero(ground_preds[idx])[0]]
    labels.append([weather_label] + ground_labels)
  return labels

def eval_once(saver, metric_ops, pred_ops):
  """Run Eval once. """
  with tf.Session() as sess:
    ckpt, global_step = restore_checkpoint(saver, sess)

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      # num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      # true_count = 0  # Counts the number of correct predictions.
      # total_sample_count = num_iter * FLAGS.batch_size
      # step = 0

      # It's not clear when the coordinator stops... This runs for many iterations
      while not coord.should_stop():
        def summary_str(op_values):
            return ', '.join(['{} = {:.3f}'.format(key, val) for key, val in op_values.items()])

        # Add names to the various ops
        metric_values = sess.run(metric_ops.values())
        print(summary_str({metric: val for metric, val in zip(metric_ops.keys(), metric_values)}))

        weather_preds, ground_preds = sess.run(pred_ops)
        predicted_labels = get_labels_from_predictions(weather_preds, ground_preds)
        for idx, labels in enumerate(predicted_labels[:20]):
          print(idx, ' '.join(labels))

        break
        
      # summary = tf.Summary()
      # summary.ParseFromString(sess.run(summary_op))
      # summary.value.add(tag='Precision @ 1', simple_value=precision)
      # summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate(filename, params):
  """ Evaluate the model """
  with tf.Graph().as_default() as g:
    images, labels = inputs(filename, batch_size=500, num_epochs=None)
    logits = inference(images)

    metric_ops = get_metric_ops(logits, labels, cutoff=params['cutoff'])
    pred_ops = get_prediction_ops(logits, labels, cutoff=params['cutoff'])

    # Restore the variables for evaluation
    saver = tf.train.Saver(tf.global_variables())

    # Build the summary operation based on the TF collection of Summaries.
    # summary_op = tf.merge_all_summaries()
    # summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, g)

    # The original cifar10 code runs multiple evaluation steps -- figure out why?
    eval_once(saver, metric_ops, pred_ops)  # , summary_writer, summary_op)


def get_filename(dataset, start_idx, end_idx):
  """ Get filename with given start, end indices. No filename validation. """
  assert dataset in ['train', 'test', 'validation'], \
         "dataset must be one of train, test, validation"
  return os.path.join(DATA_DIR, 'protobuf', 
                      'images.{}.{}_{}.proto'.format(dataset, start_idx, end_idx))

def main(_):
  filenames = {
    'train': get_filename('train', 0, 1999),
    'validation': get_filename('validation', 2000, 2499),
    'test': get_filename('test', 2500, 2999)
  }

  params = {
    'cutoff': 0.5
  }

  print("Training")
  # run_training(filenames['train'], params)

  print("Evaluating on test data:")
  evaluate(filenames['test'], params)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
