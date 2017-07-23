""" Keras strawman to predict the weather label using a single batch of numpy images """
import os
from datetime import datetime
import pytz
import numpy as np
import tensorflow as tf
import numpy as np
from keras import models, layers, losses, metrics, callbacks
from keras import backend as K

from planet_utils import KagglePlanetImage as Image, KagglePlanetImageLabels as ImageLabels, DATA_DIR

MODEL_DIR = os.path.join(DATA_DIR, 'models')
CHECKPOINT_DIR = os.path.join(MODEL_DIR, 'checkpoints')

BATCH_SIZE = 128

N_WEATHER_LABELS = len(ImageLabels.LABELS['weather'])
N_GROUND_LABELS = len(ImageLabels.LABELS['ground'])
N_LABELS = N_WEATHER_LABELS + N_GROUND_LABELS

def generate_images_and_labels(start_index, end_index=Image.NUM_TRAIN_IMAGES, batch_size=BATCH_SIZE,
                               shuffle=True):
    " A generator that yields a batch of images and labels. The generator must loop indefinitely. "
    def generate_next_image():
        r = np.arange(start_index, end_index)
        while True:
            perm = np.random.permutation(r) if shuffle else r
            for idx in perm:
                try:
                    image = Image(idx)
                except Image.ImageError:  # Skip bad images
                    continue
                yield image

    image_generator = generate_next_image()

    while True:
        images = np.zeros((batch_size, Image.HEIGHT, Image.WIDTH, Image.DEPTH))
        labels = np.zeros((batch_size, N_LABELS))
    
        for array_idx in range(batch_size):
            image = image_generator.next()
            images[array_idx] = image.image
            labels[array_idx, :N_WEATHER_LABELS] = image.labels['weather']
            labels[array_idx, N_WEATHER_LABELS:] = image.labels['ground']

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

def fbeta(y_true, y_pred, beta=2, threshold_shift=0):
    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall)

def f2_metric(y_true, y_pred):
    return fbeta(y_true, y_pred, beta=2)

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
    model_id = 'strawman'
    input_shape = (Image.HEIGHT, Image.WIDTH, Image.DEPTH)
    pooling_layer = layers.pooling.AveragePooling2D(pool_size=(16, 16), input_shape=input_shape)
    flattening_layer = layers.Flatten()
    dense_layer = layers.Dense(N_LABELS, activation='softmax')
    model = models.Sequential([pooling_layer, flattening_layer, dense_layer])

    model.compile(optimizer='adam',
                  loss=loss,
                  metrics=[weather_accuracy_metric, f2_metric])

    return model, model_id

def build_model_2():
    """"slightly less strawman model"""
    model_id = 'conv_1'
    input_shape = (Image.HEIGHT, Image.WIDTH, Image.DEPTH)
    model = models.Sequential()
    #going from 256x256x4 to 64x64x32
    model.add(layers.Conv2D(filters=32, kernel_size = (8, 8), strides=(4, 4), input_shape=input_shape, padding='valid', activation='relu'))
    # model.add(layers.Conv2D(filters=32, kernel_size = (16, 16, 32), strides=(2, 2), padding='valid', activation='relu'))
    model.add(layers.pooling.MaxPooling2D(pool_size=(4, 4)))    
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), padding='valid', activation='relu'))
    model.add(layers.pooling.MaxPooling2D(pool_size=(4, 4)))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(N_LABELS, activation='softmax'))

    model.compile(optimizer='adam',
                  loss=loss,
                  metrics=[weather_accuracy_metric, f2_metric])

    return model, model_id

def setup_callbacks(filename_head):
    """ Set up checkpoint callback """
    filepath = os.path.join(CHECKPOINT_DIR, filename_head + '_{epoch:02d}_chkpt.hdf5')
    checkpoint = callbacks.ModelCheckpoint(filepath)
    callback_list = [checkpoint]
    return callback_list

def main():
    """ Build model and evaluate it on training and test data """
    # Seed the random generator for reproducibility
    timestamp = '{:%Y-%m-%d-%H:%M:%S}'.format(datetime.now(pytz.utc))
    np.random.seed(1233)

    print "Building model..."
    model, model_id = build_model_2()
    model_filename_head = '{}_{}'.format(model_id, timestamp)
    model_filepath = os.path.join(MODEL_DIR, model_filename_head + '.hdf5')
    callback_list = setup_callbacks(model_filename_head)  

    print "Training model..."
    n_train_samples = 32000
    n_test_samples = 8000

    training_data_generator = generate_images_and_labels(0, n_train_samples)
    model.fit_generator(training_data_generator, n_train_samples / BATCH_SIZE, epochs=5, callbacks=callback_list)
    model.save(model_filepath)

    print "Evaluating..."
    training_data_generator = generate_images_and_labels(0, n_train_samples, shuffle=False)
    # one or two of training images may sneak into the test set because we skip bad images
    test_data_generator = generate_images_and_labels(n_train_samples, n_train_samples + n_test_samples,
                                                     shuffle=False)

    in_sample_score = model.evaluate_generator(training_data_generator, n_train_samples / BATCH_SIZE)
    print "In sample: {}".format(in_sample_score)

    oos_score = model.evaluate_generator(test_data_generator, n_test_samples / BATCH_SIZE)
    print "Out of sample: {}".format(oos_score)


if __name__ == '__main__':
	main()