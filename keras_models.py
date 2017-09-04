from keras import models, layers
from keras import backend as K

from planet_utils import KagglePlanetImage as Image, KagglePlanetImageLabels as ImageLabels

N_WEATHER_LABELS = len(ImageLabels.LABELS['weather'])
N_GROUND_LABELS = len(ImageLabels.LABELS['ground'])
N_LABELS = N_WEATHER_LABELS + N_GROUND_LABELS

class ModelBuilder(object):

    MODELS = {
        'strawman': 'build_strawman_model',
        'simple_conv': 'build_simple_conv_model',
        'dilated_conv': 'build_dilated_conv_model',
        'combined_conv': 'build_combined_conv_model'
    }

    def __init__(self, model_id, **kwargs):
        """ Initialize with model id. Use Kwargs for model compilation """
        if model_id not in self.MODELS:
            raise ValueError("Invalid model id '{}'".format(model_id))
        self.model_id = model_id
        self.compile_args = kwargs

    def build_model(self):
        build_method = getattr(self, self.MODELS[self.model_id])
        model = build_method()
        model.compile(**self.compile_args)
        return model

    def build_strawman_model(self):
        """ Build and compile the strawman model, which includes a pooling layer and a dense layer """
        input_shape = (Image.HEIGHT, Image.WIDTH, Image.DEPTH)
        pooling_layer = layers.pooling.AveragePooling2D(pool_size=(16, 16), input_shape=input_shape)
        flattening_layer = layers.Flatten()
        dense_layer = layers.Dense(N_LABELS, activation='softmax')
        model = models.Sequential([pooling_layer, flattening_layer, dense_layer])
        return model

    def build_simple_conv_model(self):
        """"slightly less strawman model"""
        input_shape = (Image.HEIGHT, Image.WIDTH, Image.DEPTH)
        model = models.Sequential()
        #going from 256x256x4 to 64x64x32
        model.add(layers.Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), input_shape=input_shape, padding='valid', activation='relu'))
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
        return model

    def build_dilated_conv_model(self):
        """"simple dilated convolution model"""
        input_shape = (Image.HEIGHT, Image.WIDTH, Image.DEPTH)
        model = models.Sequential()
        #going from 256x256x4 to 64x64x32
        model.add(layers.Conv2D(filters=32, kernel_size = (4, 4), dilation_rate=(4, 4), input_shape=input_shape, padding='valid', activation='relu'))
        # model.add(layers.Conv2D(filters=32, kernel_size = (16, 16, 32), strides=(2, 2), padding='valid', activation='relu'))
        model.add(layers.pooling.MaxPooling2D(pool_size=(4, 4)))    
        model.add(layers.Dropout(0.25))

        model.add(layers.Conv2D(filters=32, kernel_size=(4, 4), dilation_rate=(2, 2), padding='valid', activation='relu'))
        model.add(layers.pooling.MaxPooling2D(pool_size=(4, 4)))
        model.add(layers.Dropout(0.25))
        
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.25))
        model.add(layers.Dense(N_LABELS, activation='softmax'))
        return model

    def build_combined_conv_model(self):
        """the above two models combined"""
        input_shape = (Image.HEIGHT, Image.WIDTH, Image.DEPTH)
        image_input = layers.Input(input_shape)

        #traditional convolution
        conv1 = layers.Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), input_shape=input_shape, padding='valid', activation='relu')(image_input)
        conv1 = layers.pooling.MaxPooling2D(pool_size=(4, 4))(conv1)
        conv1 = layers.Dropout(0.25)(conv1)
        #second set
        conv1 = layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), padding='valid', activation='relu')(conv1)
        conv1 = layers.pooling.MaxPooling2D(pool_size=(4, 4))(conv1)
        conv1 = layers.Dropout(0.25)(conv1)
        #finish with flattening
        conv1 = layers.Flatten()(conv1)
        conv1 = layers.Dense(128, activation='relu')(conv1)

        #dilated convolution
        dila1 = layers.Conv2D(filters=32, kernel_size=(4, 4), dilation_rate=(4, 4), input_shape=input_shape, padding='valid', activation='relu')(image_input)
        dila1 = layers.pooling.MaxPooling2D(pool_size=(4, 4))(dila1)
        dila1 = layers.Dropout(0.25)(dila1)
        #second set
        dila1 = layers.Conv2D(filters=32, kernel_size=(4, 4), dilation_rate=(2, 2), padding='valid', activation='relu')(dila1)
        dila1 = layers.pooling.MaxPooling2D(pool_size=(4, 4))(dila1)
        dila1 = layers.Dropout(0.25)(dila1)
        #finish with flattening
        dila1 = layers.Flatten()(dila1)
        dila1 = layers.Dense(128, activation='relu')(dila1)

        #combine
        combo1 = layers.concatenate([conv1, dila1])
        combo1 = layers.Dense(256, activation='relu')(combo1)
        combo1 = layers.Dense(N_LABELS, activation='softmax')(combo1)
        model = models.Model(inputs=image_input, outputs=combo1)
        return model
