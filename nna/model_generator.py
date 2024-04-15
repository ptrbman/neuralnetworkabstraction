#!/usr/bin/env python3
import tensorflow as tf
import random
import numpy as np
from tensorflow import keras

class ModelGenerator():
    """Contains static functions to generate (i.e., trains) a model for a particular configuration.
    """


    def linear(xs, config):
        """Computes the function defined by config for inputs xs.

        :param xs: input values for the x-variables.
        :param config: config describing function.
        :returns: the resulting sum.

        """
        sum = 0
        for i in range(config.n):
            sum += config.coefficients[i]*xs[i]
        return sum


    def create_network(layer_sizes):
        """Creates a pytorch model with layer_sizes

        :param layer_sizes: Sizes of each layer in order.
        :returns: pytorch model

        """
        model = keras.models.Sequential()

        model.add(keras.layers.InputLayer(shape=(layer_sizes[0],), name='input'))
        for idx, ls in enumerate(layer_sizes[1:-1]):
            model.add(keras.layers.Dense(ls, activation='relu', name='hidden' + str(idx)))
        model.add(keras.layers.Dense(layer_sizes[-1], name='output'))

        return model


    def gen_data(count, config):
        """Generate count data-points using config.

        :param count: number of data points to generate.
        :param config: configuration of function.
        :returns: two arrays with x- and y-values for each data-point.

        """
        data_x = []
        data_y = []

        # Always include the zero-vector
        zero_x = [0]*config.n
        data_x.append(tf.transpose(tf.convert_to_tensor(zero_x)))
        data_y.append(tf.convert_to_tensor([ModelGenerator.linear(zero_x, config)]))

        for i in range(count-1):
            xs = []
            for _ in range(config.n):
                xs.append(random.randint(0,1))
            y = ModelGenerator.linear(xs, config)
            data_x.append(tf.transpose(tf.convert_to_tensor(xs)))
            data_y.append(tf.convert_to_tensor([y]))

        return (np.array(data_x), np.array(data_y))

    def generate_model(config, layer_sizes, epochs=5, count=100000):
        """Generate and train a pytorch model based on config with layer_size as layer
            sizes. Count is the number of data-points to use for training.

        :param config: configuration for underlying function
        :param layer_sizes: sizes of layers
        :param count: number of data-points to generate for training
        :returns: pytorch model

        """
        (x_train, y_train) = ModelGenerator.gen_data(count, config)
        model = ModelGenerator.create_network(layer_sizes)

        loss_fn = tf.keras.losses.MeanSquaredError(reduction='none', name="mean_squared_error")
        model.compile(optimizer='adam',
                    loss=loss_fn,
                    metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=epochs)

        return model
