#!/usr/bin/env python3

from nna.function import LinearFunction
from nna.model_generator import ModelGenerator
from nna.torch_api import TorchAPI
from nna.utils import Utils

import pickle

def quick_test(generate=False):
    f = LinearFunction(6, 1, 20)

    nn = None

    if generate:
        torch_model = ModelGenerator.generate_model(f, [f.n, 2, 3, 1], 5, 20000)
        nn = TorchAPI.torch2nn(torch_model)

        with open('test.mdl', 'wb') as outfile:
            pickle.dump(nn, outfile)
    else:
        with open('test.mdl', 'rb') as infile:
            nn = pickle.load(infile)

    #TODO: we assume all inputs are 0 or 1
    #TODO: Do we want split to be mutating or do we copy it?
    nn.color_network()

    ub = 20
    binary_bounds = Utils.binary_bounds(nn, ub)
    iterative_bounds = Utils.iterative_bounds(nn, ub)

    print("Function:      \t", f)
    print("Input\t  Binary\t  Iterative:")
    for i, (binary, iterative) in enumerate(zip(binary_bounds, iterative_bounds)):
        print(i, "\t  ", binary, "\t  ", iterative)


quick_test(True)
