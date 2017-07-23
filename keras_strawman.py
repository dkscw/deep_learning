""" Keras strawman to predict the weather label using a single batch of numpy images """

import numpy as np
import tensorflow as tf
from keras import models, layers, losses, metrics

from planet_utils import KagglePlanetImage as Image, KagglePlanetImageLabels as ImageLabels

BATCH_SIZE = 128

N_WEATHER_LABELS = len(ImageLabels.LABELS['weather'])
N_GROUND_LABELS = len(ImageLabels.LABELS['ground'])
N_LABELS = N_WEATHER_LABELS + N_GROUND_LABELS

def read_images_and_labels(start_idx=0, n_images=BATCH_SIZE):
    """ Read images as numpy array """	
    images = np.zeros((n_images, Image.HEIGHT, Image.WIDTH, Image.DEPTH))
    labels = np.zeros((n_images, N_LABELS))

    # Ensure that we read n_images even if some of them can't be loaded, unless there are none left
    array_idx, image_idx = 0, start_idx
    while array_idx < n_images:
        try:
            image = Image(image_idx)
        except Image.ImageError:
            print "Bad image at index ", image_idx 
        else:
            images[array_idx] = image.image
            labels[array_idx] = image.labels['weather']
            array_idx += 1
        image_idx += 1

    return images, labels, image_idx


def generate_images_and_labels(start_index, end_index=Image.NUM_TRAIN_IMAGES, batch_size=BATCH_SIZE,
                               shuffle=True):
    " A generator that yields a batch of images and labels. The generator must loop indefinitely. "
    def restart():
        index = start_index
        r = np.arange(start_index, end_index)
        perm = np.random.permutation(r) if shuffle else r
        return index, perm

    image_idx, perm = restart()

    while True:
        images = np.zeros((batch_size, Image.HEIGHT, Image.WIDTH, Image.DEPTH))
        labels = np.zeros((batch_size, N_LABELS))
    
        array_idx = 0
        while array_idx < batch_size:
            try:
                image = Image(perm[image_idx])
            except Image.ImageError:
                pass
            else:
                images[array_idx] = image.image
                labels[array_idx, :N_WEATHER_LABELS] = image.labels['weather']
                labels[array_idx, N_WEATHER_LABELS:] = image.labels['ground']
                array_idx += 1
            image_idx += 1
            if image_idx == end_index:
                image_idx, perm = restart()

        yield images, labels


def loss(y_true, y_pred):
    """ Build the loss function given true labels and predicted probabilities.
    Weather labels are mutually exclusive so we use categorical cross entropy ("cross entropy with
    softmax" in tf terminology). Ground labels are multi-class, so we use binary cross entropy
    ("sigmoid" in ternsorflow). Weather labels always come first. """
    return losses.categorical_crossentropy(y_true[:, :N_WEATHER_LABELS], y_pred[:, :N_WEATHER_LABELS]) + \
           losses.binary_crossentropy(y_true[:, N_WEATHER_LABELS:], y_pred[:, N_WEATHER_LABELS:])
    # # weather_true = tf.slice(y_true, [0, 0], [-1, N_WEATHER_LABELS])
    # # weather_pred = tf.slice(y_pred, [0, 0], [-1, N_WEATHER_LABELS])
    # # ground_true = tf.slice(y_true, [0, N_WEATHER_LABELS], [-1, -1])
    # # ground_pred = tf.slice(y_pred, [0, N_WEATHER_LABELS], [-1, -1])
    
    # return losses.categorical_crossentropy(weather_true, weather_pred) + \
    #        losses.binary_crossentropy(ground_true, ground_pred)

def weather_accuracy_metric(y_true, y_pred):
    return metrics.categorical_accuracy(y_true[:, :N_WEATHER_LABELS], y_pred[:, :N_WEATHER_LABELS])

# def f2_metric(y_true, y_pred):
#     """ Return list of metrics """
# def metrics(logits, labels, cutoff=0.5):
#   """ Build metric ops with uniform cutoff """
#   # we have multiple label classes (weather/common/special) but the metrics apply to all labels
#   # together. Other types are multilabeled.
#   # TODO: Replace with metrics from tf.conrib.metrics 
#   def confusion_matrix(preds, labels):
#     "Count true positives, true negatives, false positives and false negatives"
#     tp = tf.count_nonzero(preds * labels, axis=1)
#     fp = tf.count_nonzero(preds * (1 - labels), axis=1)
#     tn = tf.count_nonzero((1 - preds) * (1 - labels), axis=1)
#     fn = tf.count_nonzero((1 - preds) * labels, axis=1)
#     return tp, fp, tn, fn

#   # Weather labels are mutually exclusive, so we pick argmax for the prediction and one-hot encode
#   # them so we can count tp, fp, etc.
#   weather_preds, ground_preds = get_prediction_ops(logits, labels, cutoff)
  
#   # One-hot encode weather preds to count tp, fp, etc. Accuracy for weather prediction is just % of TP
#   one_hot_weather_preds = tf.one_hot(weather_preds, depth=len(ImageLabels.LABELS['weather']))
#   weather_tp, weather_fp, weather_tn, weather_fn = confusion_matrix(one_hot_weather_preds, labels['weather'])
#   weather_accuracy = tf.reduce_mean(tf.cast(weather_tp, tf.float32))

#   # We currently do not predict special labels. Add common label results to the current counts.
#   common_tp, common_fp, common_tn, common_fn = confusion_matrix(ground_preds, labels['common'])
#   tp = weather_tp + common_tp
#   fp = weather_fp + common_fp
#   tn = weather_tn + common_tn
#   fn = weather_fn + common_fn
      
#   # Calculate precision, recall and F2 score.
#   precision = tp / (tp + fp)
#   recall = tp / (tp + fn)
#   f2_score = (5 * precision * recall) / (4 * precision + recall)

#   # Replace nans with zeros
#   def replace_nans(tensor):
#     return tf.where(tf.is_nan(tensor), tf.zeros_like(tensor), tensor)

#   precision = replace_nans(precision)
#   recall = replace_nans(recall)
#   f2_score = replace_nans(f2_score)

#   metrics = OrderedDict([
#     ('precision', tf.reduce_mean(precision)),
#     ('recall', tf.reduce_mean(recall)),
#     ('f2_score', tf.reduce_mean(f2_score)),
#     ('weather_accuracy', weather_accuracy),
#   ])
  
#   return metrics


def build_model():
    """ Build and compile the strawman model, which includes a pooling layer and a dense layer """
    input_shape = (Image.HEIGHT, Image.WIDTH, Image.DEPTH)
    pooling_layer = layers.pooling.AveragePooling2D(pool_size=(16, 16), input_shape=input_shape)
    flattening_layer = layers.Flatten()
    dense_layer = layers.Dense(N_LABELS, activation='softmax')
    model = models.Sequential([pooling_layer, flattening_layer, dense_layer])

    model.compile(optimizer='adam',
                  loss=loss,
                  metrics=[weather_accuracy_metric])

    return model

def build_model_2():
    """"slightly less strawman model"""
    input_shape = (Image.HEIGHT, Image.WIDTH, Image.DEPTH)
    model = Sequential()
    #going from 256x256x4 to 64x64x32
    model.add(layers.Conv2D(filters=32, kernel_size = (16, 16, 4), strides=(1, 1), input_shape=input_shape, padding='valid', activation='relu'))
    model.add(layers.Conv2D(filters=32, kernel_size = (16, 16, 32), strides=(2, 2), padding='valid', activation='relu'))
    model.add(layers.pooling.MaxPooling2D(pool_size=(2,2)))    
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(filters=32, kernel_size=(16,16,32), strides=(1, 1), padding='valid', activation='relu'))
    model.add(layers.pooling.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(N_LABELS, activation='softmax'))

    model.compile(optimizer='adam',
                  loss=loss,
                  metrics=[weather_accuracy_metric])

    return model


def main():
    """ Build model and evaluate it on training and test data """
    print "Building model..."
    model = build_model()

    print "Training model..."
    n_train_samples = 32000
    n_test_samples = 8000

    training_data_generator = generate_images_and_labels(0, n_train_samples)
    model.fit_generator(training_data_generator, n_train_samples / BATCH_SIZE, epochs=5)

    print "Evaluating..."
    training_data_generator = generate_images_and_labels(0, n_train_samples, shuffle=False)
    # one or two of training images may sneak into the test set because we skip bad images
    test_data_generator = generate_images_and_labels(n_train_samples, n_train_samples + n_test_samples,
                                                     shuffle=False)

    in_sample_score = model.evaluate_generator(training_data_generator, n_train_samples / BATCH_SIZE)
    oos_score = model.evaluate_generator(test_data_generator, n_test_samples / BATCH_SIZE)
    print "In sample: {}".format(in_sample_score)
    print "Out of sample: {}".format(oos_score)


if __name__ == '__main__':
	main()