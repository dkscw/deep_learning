"""Metrics and prediction ops """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import numpy as np
from scipy.special import logit
import tensorflow as tf

from planet_utils import KagglePlanetImageLabels as ImageLabels


def get_prediction_ops(logits, labels, cutoff):
  """ Return the weather and ground predictions. """
  weather_preds = tf.argmax(logits['weather'], axis=1)
  ground_preds = tf.greater(logits['common'], logit(cutoff))
  return weather_preds, tf.cast(ground_preds, tf.float32)


def get_labels_from_predictions(weather_preds, ground_preds):
  """ Get the prediction labels. Weather predictions are the labels index; ground preds are a binary
  matrix """
  labels = []
  for idx in range(weather_preds.shape[0]):
    weather_label = ImageLabels.LABELS['weather'][weather_preds[idx]]
    ground_labels = [ImageLabels.LABELS['common'][col] for col in np.nonzero(ground_preds[idx])[0]]
    labels.append([weather_label] + ground_labels)
  return labels


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

