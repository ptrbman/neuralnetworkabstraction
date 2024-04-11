#!/usr/bin/env python3
import torch
from torch import nn
import tensorflow as tf
from nna.model_generator import ModelGenerator
from nna.neural_net import Layer, NeuralNet
import re


class TorchAPI():
    """Contains static functions for converting between NeuralNet and pytorch models.
    """

    def torch2nn(model):
        """Converts a pytorch model to a NeuralNet by extracting structure, weights and biases. There are restrictions on the network structures and naming of layers.

        :param model: pytorch model to convert
        :returns: model as a NeuralNet

        """
        def path2idx(path):
            hidden_kernel_p = r"sequential.*/hidden(\d+)/kernel"
            hidden_bias_p = r"sequential.*/hidden(\d+)/bias"
            output_kernel_p = r"sequential.*/output/kernel"
            output_bias_p = r"sequential.*/output/bias"

            match = re.search(hidden_kernel_p, v.path)
            if match:
               hidden_idx = int(match.group(1))
               return ('hidden', 'weights', hidden_idx)

            match = re.search(hidden_bias_p, v.path)
            if match:
               hidden_idx = int(match.group(1))
               return ('hidden', 'bias', hidden_idx)

            match = re.search(output_kernel_p, v.path)
            if match:
               return ('output', 'weights', 0)

            match = re.search(output_bias_p, v.path)
            if match:
               return ('output', 'bias', 0)

            raise Exception("Unhandled layer name: " + path)

        hidden_weights = {}
        hidden_biases = {}
        output_weights = None
        output_biases = None

        # We begin by extracting all layers in case they are not ordered
        for v in model.variables:
            (layer_type, data_type, layer_idx) = path2idx(v.path)
            if layer_type == 'hidden':
                if data_type == 'weights':
                    hidden_weights[layer_idx] = v.numpy().tolist()
                else:
                    hidden_biases[layer_idx] = v.numpy().tolist()
            elif layer_type == 'output':
                 if data_type == 'weights':
                    output_weights = v.numpy().tolist()
                 else:
                    output_biases = v.numpy().tolist()
            else:
                raise Exception("Unhandled layer type: " + layer_type)

        # Put all layers in order (with output layer at end)
        layers = []
        for i in range(len(hidden_weights)):
            layers.append(Layer("hidden" + str(i), len(hidden_weights[i]), len(hidden_biases[i]), hidden_weights[i], hidden_biases[i]))
        layers.append(Layer("output", len(output_weights), len(output_biases), output_weights, output_biases))

        return NeuralNet(layers)

    def nn2torch(nn):
        """Converts a NeuralNet to a pytorch model.

        :param nn: NeuralNet to convert
        :returns: A pytorch model of nn

        """
        sizes = nn.layer_sizes()
        model = ModelGenerator.create_network(nn.layer_sizes())

        # -1 as input layer has no weights
        for i in range(len(sizes)-1):
            model.layers[i].set_weights([tf.constant(nn.getWeights(i)), tf.constant(nn.getBias(i))])
        return model
