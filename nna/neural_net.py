#!/usr/bin/env python3
import tensorflow as tf
from tensorflow import keras
import torch
import numpy as np
import copy


# For a layer we store the incoming weights and biases. inputs are the number of
# nodes in the preceding layer.
class Layer:
    """Represents a layer with its incoming weights in a NeuralNet."""

    COLORLESS = 0
    GREEN = 1
    RED = -1

    def __init__(self, name, inputs, nodes, weights, biases, colors=None):
        """Constructor of Layer.

        :param name: name of layer
        :param inputs: number of inputs to layer
        :param nodes: number of nodes in layer
        :param weights: incoming weights
        :param biases: biases of nodes
        :param colors: colors of nodes (defaults to COLORLESS)
        :returns: a new Layer

        """
        self.name = name
        self.weights = weights
        self.biases = biases
        self.inputs = inputs
        self.nodes = nodes
        if colors:
            self.colors = colors
        else:
            self.colors = [Layer.COLORLESS] * self.nodes

        # Ensure proper dimensions
        assert len(weights) == inputs
        for ws in weights:
            assert len(ws) == nodes
        assert len(biases) == nodes

    def updateColors(self, newColors):
        """Update colors using a list (often obtained from getInputColors from next layer)

        :param newColors: list of new colors
        :returns: None

        """
        assert len(newColors) == len(self.colors)
        self.colors = newColors

    def getInputColors(self):
        """This propagates the colors from the current layer to the one before

        :returns: colors for the preceding layer (with possibility of COLORLESS nodes).

        """

        # If there is at least one colorless node, we can not do anything
        if Layer.COLORLESS in self.colors:
            return None

        input_colors = []

        # self.weights[0] contains the weights from input node 1
        for i in range(self.inputs):
            is_red = False
            is_green = False
            for idx, w in enumerate(self.weights[i]):
                if w > 0:
                    if self.colors[idx] == Layer.GREEN:
                        is_green = True
                    else:
                        assert self.colors[idx] == Layer.RED
                        is_red = True

                if w < 0:
                    if self.colors[idx] == Layer.GREEN:
                        is_red = True
                    else:
                        assert self.colors[idx] == Layer.RED
                        is_green = True

            if is_red and is_green:
                input_colors.append(Layer.COLORLESS)
            elif is_red:
                input_colors.append(Layer.RED)
            elif is_green:
                input_colors.append(Layer.GREEN)
            else:
                raise Exception(
                    "Unexpected coloring error (maybe all weights are zero?)"
                )
        return input_colors

    def __str__(self):
        """String representation of a Layer.

        :returns: string representation

        """
        return (
            "<<<Layer: "
            + self.name
            + ">>>\nWeights: "
            + str(self.weights)
            + "\nBias: "
            + str(self.biases)
            + "\nColors: "
            + str(self.colors)
        )


class NeuralNet:
    """Represents a Neural Network which can be manipulated via coloring, split and
    creating difference networks.

    """

    def __init__(self, layers):
        """NeuralNet constructor.

        :param layers: layers of the network
        :returns: the network

        """
        self.layers = layers

    def getWeights(self, layer_idx):
        """Get weights of layer layer_idx.

        :param layer_idx: layer index
        :returns: weights of layer_idx

        """
        return self.layers[layer_idx].weights

    def getBias(self, layer_idx):
        """Get bias of layer layer_idx.

        :param layer_idx: layer index
        :returns: bias of layer_idx

        """
        return self.layers[layer_idx].biases

    # TODO: Do we want this to be in-place or copy?
    def split_node(self, layer, node):
        """Splits node in layer into two nodes, obtaining an equivalent network.
        We need to do three things:

        1) Replace node in layer by two nodes node_inc and node_dec. node_inc
        should only point with positive (negative) weights to green (red) nodes
        and node_dec should only point with positive (negative) weights to red
        (green) nodes)

        2) Update incoming weights. We need to make a copy for each incoming
        weight to node such that it there are two copies, one to node_inc and
        one to node_dec.

        3) Update outgoing weights. Node_inc should keep all weights pointing
        towards positive (negative) green (red) nodes, the rest should be zero.
        Node_inc should keep all weights pointing towards positive (negative)
        red (green) nodes, the rest should be zero. This must be done for
        layer+1

        :param layer: layer to split
        :param node: node to split
        :returns: None

        """

        # TODO: can we show that we only need two colors?

        # Current Layer
        cur_layer = self.layers[layer]
        cl_name = cur_layer.name
        cl_inputs = cur_layer.inputs
        cl_nodes = cur_layer.nodes + 1
        cl_biases = []

        cl_weights = []
        for w in cur_layer.weights:
            cl_weights.append(w[0:node] + [w[node]] * 2 + w[node + 1 :])

        cl_biases = (
            cur_layer.biases[0:node]
            + [cur_layer.biases[node]] * 2
            + cur_layer.biases[node + 1 :]
        )

        new_cur_layer = Layer(cl_name, cl_inputs, cl_nodes, cl_weights, cl_biases)
        self.layers[layer] = new_cur_layer

        # If this is not the last layer, update following layer
        if len(self.layers) > layer + 1:
            next_layer = self.layers[layer + 1]
            nl_weights = []

            for cl_node, cl_node_ws in enumerate(next_layer.weights):
                if cl_node == node:
                    inc_weights = []
                    dec_weights = []
                    for color, w in zip(next_layer.colors, cl_node_ws):
                        if (color == Layer.GREEN and w > 0) or (
                            color == Layer.RED and w < 0
                        ):
                            inc_weights.append(w)
                            dec_weights.append(0)
                        elif (color == Layer.RED and w > 0) or (
                            color == Layer.GREEN and w < 0
                        ):
                            inc_weights.append(0)
                            dec_weights.append(w)
                        elif color == Layer.COLORLESS:
                            raise Exception("Trying to split to COLORLESS node")
                        elif w == 0:
                            raise Exception("Cannot handle zero weights")

                    nl_weights.append(inc_weights)
                    nl_weights.append(dec_weights)
                else:
                    nl_weights.append(cl_node_ws)

            nl_name = next_layer.name
            nl_inputs = next_layer.inputs + 1
            nl_nodes = next_layer.nodes
            nl_biases = next_layer.biases
            nl_colors = next_layer.colors

            self.layers[layer + 1] = Layer(
                nl_name, nl_inputs, nl_nodes, nl_weights, nl_biases, nl_colors
            )

    def color_network(self):
        """Colors network in place. We try to color the next layer If it doesn't work, we need to split
        some nodes ... The nodes to be split are in the "current" layer
        (i.e., the one we failed to color) We go through the nodes one by one
        and see if it needs splitting and form a new list of nodes. Then we
        must update both this and the next layers weights.

        :returns: None

        """

        # TODO: Assumption that network has one output node
        # Color output node
        self.layers[-1].updateColors([Layer.GREEN])

        # Begin with layer before output layer and iterate until first layer
        cur_layer = len(self.layers) - 2
        while cur_layer >= 0:
            colors = self.layers[cur_layer + 1].getInputColors()

            # This means we have a colorless node, we need to split it!
            # If we do it in reverse order, the indices will not change
            if Layer.COLORLESS in colors:
                loop = list(enumerate(colors))
                loop.reverse()
                for idx, color in loop:
                    if color == Layer.COLORLESS:
                        self.split_node(cur_layer, idx)
            else:
                self.layers[cur_layer].updateColors(colors)
                cur_layer -= 1

    def layer_sizes(self):
        """Returns the sizes of the layers in the network (including the input layer).

        :returns: Layer sizes.

        """
        internal_layers = list(map(lambda x: x.nodes, self.layers))
        return [self.layers[0].inputs] + internal_layers

    def __str__(self):
        """String representation of NeuralNet

        :returns: string representation

        """
        lines = ["Neural Network"]
        for l in self.layers:
            lines.append(str(l))
        return "\n".join(lines)

    def remove_input(self, input_node, maximizing, MAX_IN=1):
        """Functions for removing a node and creating a difference network and verifying
        a bound Remove node from list, and replace with MAX case if maximizing
        otherwise MIN.

        :param input_node: input_node to remove
        :param maximizing: True if maximizing network should be created
        :returns: a new maximizing/minimizing network

        """
        first_layer = self.layers[0]
        fl_weights = (
            self.layers[0].weights[0:input_node]
            + self.layers[0].weights[input_node + 1 :]
        )

        fl_biases = []
        for i in range(first_layer.nodes):
            b = first_layer.biases[i]
            w = first_layer.weights[input_node][i]
            c = first_layer.colors[i]

            # Here we assume that MAX = 1 (thus b += w instead of b += MAX*w)
            if maximizing:
                if (w > 0 and c == Layer.GREEN) or (w < 0 and c == Layer.RED):
                    b += w * MAX_IN
            else:
                if (w > 0 and c == Layer.RED) or (w < 0 and c == Layer.GREEN):
                    b += w * MAX_IN

            # Also, if we have non-zero minimum, we might have to do b += MIN*w (or b-= MIN*w)

            fl_biases.append(b)

        fl_name = first_layer.name
        fl_inputs = first_layer.inputs - 1
        fl_nodes = first_layer.nodes
        new_first_layer = Layer(fl_name, fl_inputs, fl_nodes, fl_weights, fl_biases)

        new_network = copy.deepcopy(self)
        new_network.layers[0] = new_first_layer
        return new_network

    def merge_input_layer(l1, l2):
        """Merges two input layers l1 and l2. When merging input layers, the resulting
        layer should share the same inputs.

        :param l1: first layer to merge
        :param l2: second layer to merge
        :returns: a layer which is the merging of l1 and l2

        """
        name = l1.name + "." + l2.name
        inputs = l1.inputs
        assert l1.inputs == l2.inputs
        nodes = l1.nodes + l2.nodes
        weights = []
        for w1, w2 in zip(l1.weights, l2.weights):
            weights.append(w1 + w2)
        biases = l1.biases + l2.biases

        return Layer(name, inputs, nodes, weights, biases)

    def merge_layers(l1, l2):
        """Merges two internal layers of l1 and l2. When merging internal layers, the
        top half of the resulting layer should only receive the top half of the inputs
        (and resp. for bottom half).

        :param l1: first layer to merge
        :param l2: second layer to merge
        :returns: a layer which is the merging of l1 and l2

        """

        name = l1.name + "." + l2.name
        inputs = l1.inputs + l2.inputs
        nodes = l1.nodes + l2.nodes
        weights = []
        for w in l1.weights:
            weights.append(w + [0] * l1.nodes)
        for w in l2.weights:
            weights.append([0] * l2.nodes + w)

        biases = l1.biases + l2.biases

        return Layer(name, inputs, nodes, weights, biases)

    # TODO: Add assertion that network is colored
    def create_difference_network(self, input_node, MAX_IN):
        """Creates difference network obtained by removing input_note. This is a network
        which as an output obtains the difference between the over-estimation and
        under-estimation of removing input_node.

        :param input_node: node to remove
        :returns: a difference network

        """
        network_inc = self.remove_input(input_node, True, MAX_IN)
        network_dec = self.remove_input(input_node, False, MAX_IN)

        # We have colors of the first hidden layer. Go through the inputs, keep
        # everything except input_node. When removing input_node, we create two
        # new networks, one maximizing and one minimizing. Maximizing network
        # adds to bias such that the end result is maximized, while the
        # minimizing network adds to bias such that end result is minimzed.

        # First layer is special as it has inputs which should be duplicated:
        layers = [
            NeuralNet.merge_input_layer(network_inc.layers[0], network_dec.layers[0])
        ]

        # We merge all internal layers:
        for idx, (inc_l, dec_l) in enumerate(
            zip(network_inc.layers[1:], network_dec.layers[1:])
        ):
            layers.append(NeuralNet.merge_layers(inc_l, dec_l))

        subtraction_layer = []
        name = "subtraction"
        inputs = 2
        nodes = 1
        weights = [[1], [-1]]
        biases = [0]
        sub_layer = Layer(name, inputs, nodes, weights, biases)
        layers.append(sub_layer)

        return NeuralNet(layers)
