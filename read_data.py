# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for reading Kaggle-Planet data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed

from utils import KagglePlanetImage as Image, KagglePlanetImageLabels as ImageLabels


def extract_images_and_labels(start=0, end=Image.NUM_TRAIN_IMAGES):
  """Extract the images into a 4D float32 numpy array [index, y, x, depth].
  Also extracts the labels. """
  images = numpy.zeros((end - start, Image.HEIGHT, Image.WIDTH, Image.DEPTH))
  labels = numpy.zeros((end - start, len(ImageLabels.COMMON_LABELS)))
  for idx in range(start, end):
    image = Image(idx)
    images[idx], labels[idx] = image.image, image.label_array
  
  return images.astype(numpy.float32), labels.astype(numpy.uint8)


class DataSet(object):

  def __init__(self,
               images,
               labels,
               reshape=True,
               seed=None):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    numpy.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtypes.float32).base_dtype
    assert images.shape[0] == labels.shape[0], (
        'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
    self._num_examples = images.shape[0]

    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns*depth]
    if reshape:
      images = images.reshape(images.shape[0], Image.SIZE)

    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return numpy.concatenate((images_rest_part, images_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]


def read_data_sets(reshape=True, seed=None):
  train_images, train_labels = extract_images_and_labels(start=0, end=1000)
  train = DataSet(
      train_images, train_labels, reshape=reshape, seed=seed)
  return base.Datasets(train=train, validation=None, test=None)

  # For now ignore test and validation sets, keeping original code below for future reference
  # local_file = base.maybe_download(TEST_IMAGES, train_dir,
  #                                  SOURCE_URL + TEST_IMAGES)
  # with open(local_file, 'rb') as f:
  #   test_images = extract_images(f)

  # local_file = base.maybe_download(TEST_LABELS, train_dir,
  #                                  SOURCE_URL + TEST_LABELS)
  # with open(local_file, 'rb') as f:
  #   test_labels = extract_labels(f, one_hot=one_hot)

  # if not 0 <= validation_size <= len(train_images):
  #   raise ValueError(
  #       'Validation size should be between 0 and {}. Received: {}.'
  #       .format(len(train_images), validation_size))

  # validation_images = train_images[:validation_size]
  # validation_labels = train_labels[:validation_size]
  # train_images = train_images[validation_size:]
  # train_labels = train_labels[validation_size:]

  # train = DataSet(
  #     train_images, train_labels, dtype=dtype, reshape=reshape, seed=seed)
  # validation = DataSet(
  #     validation_images,
  #     validation_labels,
  #     dtype=dtype,
  #     reshape=reshape,
  #     seed=seed)
  # test = DataSet(
  #     test_images, test_labels, dtype=dtype, reshape=reshape, seed=seed)

  # return base.Datasets(train=train , validation=validation, test=test)
