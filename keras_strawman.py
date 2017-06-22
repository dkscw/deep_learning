""" Keras strawman to predict the weather label using a single batch of numpy images """

import numpy as np
from keras import models, layers

from planet_utils import KagglePlanetImage as Image, KagglePlanetImageLabels as ImageLabels

BATCH_SIZE = 128
N_WEATHER_LABELS = len(ImageLabels.LABELS['weather'])

def read_images_and_labels(start_idx = 0, n_images=BATCH_SIZE):
    """ Read images as numpy array """	
    images = np.zeros((n_images, Image.HEIGHT, Image.WIDTH, Image.DEPTH))
    labels = np.zeros((n_images, N_WEATHER_LABELS))

    for array_idx, image_idx in enumerate(range(start_idx, start_idx + n_images)):
        image = Image(image_idx)
        images[array_idx] = image.image
        labels[array_idx] = image.labels['weather']

    return images, labels


def generate_images_and_labels(start_index, end_index, batch_size=BATCH_SIZE):
    """ A generator that yields a batch of images and labels """
    index = start_index
    while index < end_index:
        images, labels = read_images_and_labels(index, batch_size)
        yield images, labels
        index += batch_size


def build_model():
    """ Build and compile the strawman model, which includes a pooling layer and a dense layer """
    input_shape = (Image.HEIGHT, Image.WIDTH, Image.DEPTH)
    pooling_layer = layers.pooling.AveragePooling2D(pool_size=(8, 8), input_shape=input_shape)
    flattening_layer = layers.Flatten()
    dense_layer = layers.Dense(N_WEATHER_LABELS, activation='softmax')
    model = models.Sequential([pooling_layer, flattening_layer, dense_layer])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    return model


def main():
    """ Build model and evaluate it on training and test data """
    print "Building model..."
    model = build_model()

    print "Training model..."
    n_train_samples = 32000
    n_test_samples = 8000

    training_data_generator = generate_images_and_labels(0, n_train_samples, BATCH_SIZE)
    model.fit_generator(training_data_generator, n_train_samples / BATCH_SIZE, epochs=5)

    print "Evaluating..."
    training_data_generator = generate_images_and_labels(0, n_train_samples, BATCH_SIZE)
    test_data_generator = generate_images_and_labels(n_train_samples, n_train_samples + n_test_samples,
                                                     BATCH_SIZE)

    in_sample_score = model.evaluate_generator(training_data_generator, n_train_samples / BATCH_SIZE)
    oos_score = model.evaluate_generator(test_data_generator, n_test_samples / BATCH_SIZE)
    print "In sample: {}".format(in_sample_score)
    print "Out of sample: {}".format(oos_score)


if __name__ == '__main__':
	main()