""" Keras strawman to predict the weather label using a single batch of numpy images """
import os
from datetime import datetime
import pytz
import argparse
import numpy as np
import tensorflow as tf
import numpy as np
from keras import models, layers, losses, metrics, callbacks
from keras import backend as K

from planet_utils import KagglePlanetImage as Image, KagglePlanetImageLabels as ImageLabels, DATA_DIR
from keras_models import ModelBuilder

MODEL_DIR = os.path.join(DATA_DIR, 'models')
CHECKPOINT_DIR = os.path.join(MODEL_DIR, 'checkpoints')
LOGS_DIR = os.path.join(MODEL_DIR, 'logs')

BATCH_SIZE = 100

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


def weather_accuracy_metric(y_true, y_pred):
    return metrics.categorical_accuracy(y_true[:, :N_WEATHER_LABELS], y_pred[:, :N_WEATHER_LABELS])


def fbeta(y_true, y_pred, beta=2, threshold_shift=np.zeros(N_GROUND_LABELS)):
    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # change this to use argmax for weather where there's one true label
    w_indices = K.argmax(y_pred[:, :N_WEATHER_LABELS], 1)
    y_pred_bin_w = K.one_hot(w_indices, N_WEATHER_LABELS)  # second parameter is depth
    # and just round for ground
    y_pred_bin_g = K.round(y_pred[:,N_WEATHER_LABELS:] + threshold_shift)
    # combine
    y_pred_bin = K.concatenate([y_pred_bin_w, y_pred_bin_g], 1)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall)

def f2_metric(y_true, y_pred):
    return fbeta(y_true, y_pred, beta=2)

def numpy_fbeta(y_true, y_pred, beta=2, threshold_shift=np.zeros(N_GROUND_LABELS)):
    """ Same as fbeta but operates on numpy arrays instead of tensors """
    y_pred = np.clip(y_pred, 0, 1)
    w_indices = np.argmax(y_pred[:, :N_WEATHER_LABELS], 1)
    y_pred_bin_w = np.zeros([y_pred.shape[0], N_WEATHER_LABELS])
    y_pred_bin_w[np.arange(y_pred.shape[0]), w_indices] = 1.
    # and just round for ground
    y_pred_bin_g = np.round(y_pred[:, N_WEATHER_LABELS:] + threshold_shift)
    # combine
    y_pred_bin = np.concatenate([y_pred_bin_w, y_pred_bin_g], 1)

    tp = np.sum(np.round(y_true * y_pred_bin))
    fp = np.sum(np.round(np.clip(y_pred_bin - y_true, 0, 1)))
    fn = np.sum(np.round(np.clip(y_true - y_pred, 0, 1)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall)

def tune_threshold(y_true, y_pred, beta=2):
    """ Tune the thresholds. Expects numpy arrays """
    threshold = np.zeros(N_GROUND_LABELS)
    max_score = numpy_fbeta(y_true, y_pred, beta)
    for label_idx in range(N_GROUND_LABELS):
        best_val = 0.
        for t_val in np.arange(-.5, .51, .025):
            threshold[label_idx] = t_val
            score = numpy_fbeta(y_true, y_pred, beta, threshold)
            if score > max_score:
                best_val = t_val
                max_score = score
        threshold[label_idx] = best_val

    return threshold, max_score

def setup_callbacks(filename_head):
    """ Set up checkpoint callback """
    checkpoint_filename = os.path.join(CHECKPOINT_DIR, filename_head + '_{epoch:02d}_chkpt.hdf5')
    checkpoint_callback = callbacks.ModelCheckpoint(checkpoint_filename)

    log_filename = os.path.join(LOGS_DIR, filename_head + '.log.csv')
    csv_logger_callback = callbacks.CSVLogger(log_filename, separator=',', append=False)

    return [checkpoint_callback, csv_logger_callback]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='model id', type=str, default='')
    parser.add_argument('-l', '--load', help='initial model to load', type=str, default='')
    parser.add_argument('-e', '--epochs', help='number of epochs', type=int, default=5)
    parser.add_argument('--no_eval', help='do not perform evaluation', dest='eval', action='store_false')
    parser.set_defaults(eval=True)
    return parser.parse_args()

def main():
    """ Build model and evaluate it on training and test data """
    # Seed the random generator for reproducibility
    timestamp = '{:%Y-%m-%d-%H:%M:%S}'.format(datetime.now(pytz.utc))
    np.random.seed(1233)

    print "Building model..."
    args = parse_args()
    model_id = args.model
    model = ModelBuilder(args.model, optimizer='adam', loss=loss).build_model() 
                         # loss=loss, metrics=[weather_accuracy_metric, f2_metric]).build_model()
    if args.load:
        print "Loading model from file"
        model.load_weights(os.path.join(MODEL_DIR, args.load))
    model_filename_head = '{}_{}'.format(model_id, timestamp)
    model_filepath = os.path.join(MODEL_DIR, model_filename_head + '.hdf5')
    callback_list = setup_callbacks(model_filename_head)  

    print "Training model... Model id: {}".format(model_id)
    n_train_samples = 28000
    n_validation_samples = 6000
    n_test_samples = 6000

    training_data_generator = generate_images_and_labels(0, n_train_samples)
    model.fit_generator(training_data_generator, n_train_samples / BATCH_SIZE, epochs=args.epochs, callbacks=callback_list)
    model.save(model_filepath)

    print "Evaluating..."
    if args.eval:
        training_data_generator = generate_images_and_labels(0, n_train_samples, shuffle=False)
        test_data_generator = generate_images_and_labels(n_train_samples + n_validation_samples,
                                                         n_train_samples + n_validation_samples + n_test_samples,
                                                         shuffle=False)

        # Evaluation is very slow, only compute OOS
        # in_sample_score = model.evaluate_generator(training_data_generator, n_train_samples / BATCH_SIZE)
        # print "In sample: {}".format(in_sample_score)

        oos_score = model.evaluate_generator(test_data_generator, n_test_samples / BATCH_SIZE)
        print "Out of sample: {}".format(oos_score)

    # Tune the thresholds on the validation data: calculate the thresholds for each batch and take the average
    # over all batches
    print "Tuning thresholds..."
    validation_data_generator = generate_images_and_labels(n_train_samples, n_train_samples + n_validation_samples,
                                                           shuffle=False)
    
    threshold = np.zeros(N_GROUND_LABELS)
    for step in range(n_validation_samples / BATCH_SIZE):
        images, labels = validation_data_generator.next()
        preds = model.predict_on_batch(images)
        curr_threshold, score = tune_threshold(labels, preds)
        threshold += curr_threshold
        print '{}, {}\r'.format(curr_threshold, score),
    threshold /= step
    print threshold

    new_f2_metric = lambda y_true, y_pred: fbeta(y_true, y_pred, beta=2, threshold_shift=curr_threshold)
    model.compile(optimizer='adam',
                  loss=loss,
                  metrics=[weather_accuracy_metric, new_f2_metric])
    test_data_generator = generate_images_and_labels(n_train_samples + n_validation_samples,
                                                     n_train_samples + n_validation_samples + n_test_samples,
                                                     shuffle=False)
    oos_score = model.evaluate_generator(test_data_generator, n_test_samples / BATCH_SIZE)
    print oos_score

if __name__ == '__main__':
	main()