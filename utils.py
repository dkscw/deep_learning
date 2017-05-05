"""
General utility functions for reading and processing Kaggle-Planet images
"""
import os
import csv
import skimage.io

DATA_DIR = os.environ['KAGGLE_DATA_PATH']


class ImageLoader(object):
    """ Class to load tif images and their labels. Initialize with the image sequence number. """

    ALL_LABELS = None  # Initialized below
    NUM_TRAIN_IMAGES = 40479

    def __init__(self, seqnum, is_training_image=True):
        """ Initialize with sequence number. If is_training_image is False, then read from the test
        set. There are no labels for the test set. """
        assert seqnum in range(self.NUM_TRAIN_IMAGES), \
               "Sequence number must be between 0 and {}".format(self.NUM_TRAIN_IMAGES)
        self.seqnum = seqnum
        self.is_training_image = is_training_image
        self.image = self._read_image()
        self.labels = self.ALL_LABELS[seqnum] if self.is_training_image else None

    def _read_image(self):
        """ Read the image. Returns a 256x256x4 numpy array """
        mode = 'train' if self.is_training_image else 'test'
        path = os.path.join(DATA_DIR, '{}-tif'.format(mode), '{}_{}.tif'.format(mode, self.seqnum))
        return skimage.io.imread(path)

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


ImageLoader.ALL_LABELS = ImageLoader._process_all_labels()