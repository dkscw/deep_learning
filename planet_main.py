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

from planet_utils import KagglePlanetImage as Image, KagglePlanetImageLabels as ImageLabels, DATA_DIR
from planet_input import inputs
from planet_inference import inference, loss
from planet_metrics import get_metric_ops, get_prediction_ops, get_labels_from_predictions

FLAGS = None
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'planet_train_ckpt')


def run_training(filename, params):
  """Train MNIST for a number of steps."""
  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Input images and labels. Default values for now, read from args later
    images, labels = inputs(filename, batch_size=2000, num_epochs=1)

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


def restore_checkpoint(saver, sess):
    ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
    if not ckpt or not ckpt.model_checkpoint_path: 
      raise IOError('No checkpoint file found')

    saver.restore(sess, ckpt.model_checkpoint_path)
    # Extract global_step from model_checkpoint_path, assuming it looks something like:
    # /project_path/train/model.ckpt-0
    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    return ckpt, global_step


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
  run_training(filenames['train'], params)

  print("Evaluating on test data:")
  evaluate(filenames['test'], params)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
