#!/usr/bin/env python3

from nna.function import *
from nna.model_generator import ModelGenerator
from nna.torch_api import TorchAPI
from nna.utils import Utils
import pickle
import numpy as np

nn_file = 'towers.mdl'
nn = None
def train():
    # filename = 'hitachi_data/parsed_small.csv'
    filename = 'triangulation/towers_x.csv'
    layer_sizes = [13, 7, 1]
    model = ModelGenerator.csv2model(filename, layer_sizes, 100) # 100 here is the maximum input to the network
    nn = TorchAPI.torch2nn(model)

    with open(nn_file, 'wb') as outfile:
        pickle.dump(nn, outfile)
    return model


def estimate():
    with open(nn_file, 'rb') as infile:
        nn = pickle.load(infile)


    nn.color_network()

    ub = 100 # Maximum bound when checking for bounds
    binary_bounds = Utils.binary_bounds(nn, ub, 100) # The 1 here is the maximum input to the network

    print("Input\t  Binary:")
    for i, binary in enumerate(binary_bounds):
        print(i, "\t  ", binary)

# train()
estimate()
