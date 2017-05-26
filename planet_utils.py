"""
General utility functions for reading and processing Kaggle-Planet images
"""
import os
import csv
import sys
from collections import OrderedDict

import argparse
import numpy as np
import skimage.io
import tensorflow as tf

DATA_DIR = os.environ['KAGGLE_DATA_PATH']


class TFFeature(object):
    """ Helper class to handle TF features """
    @staticmethod
    def int64_feature(value):
        """ Convert an int or a list of ints to a tf feature """
        value = value if isinstance(value, list) else [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class KagglePlanetImageLabels(object):
    """ Class to handle image labels. Only needs to be initialized once. """
    LABELS = OrderedDict([
        ('weather', ['clear',
                    'partly_cloudy',
                    'haze',
                    'cloudy']),
        ('common', ['primary',
                   'agriculture',
                   'road',
                   'water',
                   'cultivation',
                   'habitation']),
        ('special', ['bare_ground',
                    'selective_logging',
                    'artisinal_mine',
                    'blooming',
                    'slash_burn',
                    'blow_down',
                    'conventional_mine'])
    ])

    def __init__(self):
        self.all_image_labels = self._process_all_labels()

    def get_labels(self, seqnum):
        "Return the labels for the training image with the given sequence number"
        return self.all_image_labels[seqnum]

    def get_label_array(self, seqnum, label_type):
        """ Get the labels of given type as an array """
        ref_labels = self.LABELS[label_type]
        labels = self.get_labels(seqnum)
        return np.array([l in labels for l in ref_labels]).astype(np.int64)

    @staticmethod
    def _process_all_labels():
        """ Read the labels file and return a list of the labels for each image """
        labels_filename = os.path.join(DATA_DIR, 'train.csv')
        labels = []
        with open(labels_filename) as labels_csv:
            labels_reader = csv.reader(labels_csv)
            labels_reader.next()  # skip header
            for _, image_labels in labels_reader:
                labels.append(image_labels.split())
        return labels


class KagglePlanetImage(object):
    """ Class to load tif images and their labels. Initialize with the image sequence number. """

    label_processor = KagglePlanetImageLabels()
    NUM_TRAIN_IMAGES = 40479
    HEIGHT, WIDTH, DEPTH = 256, 256, 4
    SIZE = HEIGHT * WIDTH * DEPTH

    def __init__(self, seqnum, is_training_image=True):
        """ Initialize with sequence number. If is_training_image is False, then read from the test
        set. There are no labels for the test set. """
        assert seqnum in range(self.NUM_TRAIN_IMAGES), \
               "Sequence number must be between 0 and {}".format(self.NUM_TRAIN_IMAGES)
        self.seqnum = seqnum
        self.is_training_image = is_training_image
        self.image, self.jpg = self._read_image()
        self.image = self.image * 1. / self.image.max()  # Normalize
        # Labels. For now only process common labels.
        self.labels = None
        if self.is_training_image:
            self.labels = OrderedDict()
            for label_type in self.label_processor.LABELS:
                self.labels[label_type] = self.label_processor.get_label_array(seqnum, label_type)

    def _read_image(self):
        """ Read the image, both tif and jpg. The tif returns a 256x256x4 numpy array """
        mode = 'train' if self.is_training_image else 'test'
        tif_path = os.path.join(DATA_DIR, '{}-tif'.format(mode), '{}_{}.tif'.format(mode, self.seqnum))  # noqa
        jpg_path = os.path.join(DATA_DIR, '{}-jpg'.format(mode), '{}_{}.jpg'.format(mode, self.seqnum))  # noqa
        return skimage.io.imread(tif_path), skimage.io.imread(jpg_path)

    def as_feature_dict(self):
        """ Return a dict representation of the image where the values are tensorflow features """
        # Image feature
        image_byteslist = tf.train.BytesList(value=[self.image.tostring()])
        features = {'image_raw': tf.train.Feature(bytes_list=image_byteslist)}
        
        # Label features
        for label_type in self.label_processor.LABELS:
            features[label_type] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=self.labels[label_type]))

        return features

    def as_protobuf(self):
        """ Return the image as a serialized protobuf """
        return tf.train.Example(features=tf.train.Features(feature=self.as_feature_dict()))\
                       .SerializeToString()
    
    @property
    def red(self):
        return self.image[:,:,0]

    @property
    def green(self):
        return self.image[:,:,1]

    @property
    def blue(self):
        return self.image[:,:,2]

    @property
    def rgb(self):
        return self.image[:,:,:3]
    
    @property
    def nir(self):
        return self.image[:,:,3]
    
    @property
    def ndvi(self):
        return (self.nir - self.red) / (self.nir + self.red)


def serialize_batch(filename, start=0, end=KagglePlanetImage.NUM_TRAIN_IMAGES, verbose_step=500):
    """ Seriazlie a batch of images to the given filename """
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(start, end):
        if (index - start + 1) % verbose_step == 0:
            print index
        try:
            image = KagglePlanetImage(index)
            writer.write(image.as_protobuf())
        except Exception as e:
            print "Failed at index {}: {}".format(index, str(e))
            pass
    writer.close()

def parse_args():
    """ Parse command line arguments """
    parser = argparse.ArgumentParser()
  
    parser.add_argument('-s', '--start_idx', help='start index', type=int, default=0)
    parser.add_argument('-e', '--end_idx', help='end index (inclusive)', type=int,
                        default=KagglePlanetImage.NUM_TRAIN_IMAGES)
    parser.add_argument('-d', '--dataset', type=str, default='train',
                        help='data set type, default train. added to output filename')
                        
    return parser.parse_args()

if __name__ == '__main__':
    """ Example: Serialize a batch of 10000 images """
    args = parse_args()
    out_filename = 'images.{a.dataset}.{a.start_idx}_{a.end_idx}.proto'.format(a=args)
    serialize_batch(os.path.join(DATA_DIR, 'protobuf', out_filename), 
                    args.start_idx, args.end_idx + 1)  # end index is inclusive 

