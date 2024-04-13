#!/usr/bin/env python3

from nna.configuration import Configuration
from nna.model_generator import ModelGenerator
from nna.torch_api import TorchAPI
from nna.utils import Utils

import pickle

def quick_test(generate=False):
    c = Configuration(6, 1, 100)

    nn = None

    if generate:
        torch_model = ModelGenerator.generate_model(c, [c.n, 2, 3, 1])
        nn = TorchAPI.torch2nn(torch_model)

        with open('test.mdl', 'wb') as outfile:
            pickle.dump(nn, outfile)
    else:
        with open('test.mdl', 'rb') as infile:
            nn = pickle.load(infile)

    #TODO: we assume all inputs are 0 or 1
    #TODO: Do we want split to be mutating or do we copy it?
    nn.color_network()

    ub = 100
    binary_bounds = Utils.binary_bounds(nn, ub)
    iterative_bounds = Utils.iterative_bounds(nn, ub)

    print("Coefficients:      \t", c.coefficients)
    print("Input\tBinary\tIterative:")
    for i, (binary, iterative) in enumerate(zip(binary_bounds, iterative_bounds)):
        print("\t", i, "\t", binary, "\t", iterative)


quick_test(True)
