#!/usr/bin/env python3
import tensorflow as tf
import random
import numpy as np
from tensorflow import keras

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import tensorflow as tf
import pickle


class ModelGenerator:
    """Contains static functions to generate (i.e., trains) a model for a particular function."""

    def create_network(layer_sizes):
        """Creates a pytorch model with layer_sizes

        :param layer_sizes: Sizes of each layer in order.
        :returns: pytorch model

        """
        model = keras.models.Sequential()

        model.add(keras.layers.InputLayer(shape=(layer_sizes[0],), name="input"))
        for idx, ls in enumerate(layer_sizes[1:-1]):
            model.add(
                keras.layers.Dense(ls, activation="relu", name="hidden" + str(idx))
            )
        model.add(keras.layers.Dense(layer_sizes[-1], name="output"))

        return model

    def gen_data(count, function, upper_bound):
        """Generate count data-points using function.

        :param count: number of data points to generate.
        :param function: function.
        :returns: two arrays with x- and y-values for each data-point.

        """
        data_x = []
        data_y = []

        # Always include the zero-vector
        zero_x = [0] * function.n
        data_x.append(tf.transpose(tf.convert_to_tensor(zero_x)))
        data_y.append(tf.convert_to_tensor([function.evaluate(zero_x)]))

        for i in range(count - 1):
            xs = []
            for _ in range(function.n):
                xs.append(random.randint(0, upper_bound))
            y = function.evaluate(xs)
            data_x.append(tf.transpose(tf.convert_to_tensor(xs)))
            data_y.append(tf.convert_to_tensor([y]))

        return (np.array(data_x), np.array(data_y))

    def generate_model(function, layer_sizes, upper_bound, epochs=5, count=100000):
        """Generate and train a pytorch model based on function with layer_size as layer
            sizes. Count is the number of data-points to use for training.

        :param config: underlying function
        :param layer_sizes: sizes of layers
        :param count: number of data-points to generate for training
        :returns: pytorch model

        """
        (x_train, y_train) = ModelGenerator.gen_data(count, function, upper_bound)
        model = ModelGenerator.create_network(layer_sizes)

        loss_fn = tf.keras.losses.MeanSquaredError(
            reduction="none", name="mean_squared_error"
        )
        model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

        model.fit(x_train, y_train, epochs=epochs)

        return model

    def csv2model(infile, layer_sizes, epochs=50):

        df = pd.read_csv(infile, index_col=False)

        X = df[df.columns[1:]]
        y = df[df.columns[0:1]]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_val = X_train[-100:]
        y_val = y_train[-100:]
        X_train = X_train[:-100]
        y_train = y_train[:-100]

        print(X)
        print(y)

        # SMALL has 13 features
        # MEDIUM has 17 features
        torch_model = ModelGenerator.create_network(layer_sizes)
        loss_fn = tf.keras.losses.MeanSquaredError(
            reduction="none", name="mean_squared_error"
        )
        torch_model.compile(optimizer="adamw", loss=loss_fn, metrics=["accuracy"])

        _ = torch_model.fit(
            X_train,
            y_train,
            batch_size=32,
            epochs=epochs,
            # We pass some validation for
            # monitoring validation loss and metrics
            # at the end of each epoch
            validation_data=(X_val, y_val),
        )

        return torch_model
