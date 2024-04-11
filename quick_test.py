#!/usr/bin/env python3

from nna.configuration import Configuration
from nna.model_generator import ModelGenerator
from nna.torch_api import TorchAPI
import numpy as np
import tensorflow as tf
from nna.marabou_api import MarabouAPI
import pickle

def do(generate=False):
    # Configuration for data generation
    c = Configuration(4, [1,3], 1000, 1)

    new_nn = None

    if generate:
        torch_model = ModelGenerator.generate_model(c, [c.n, 2, 3, 1])
        new_nn = TorchAPI.torch2nn(torch_model)

        with open('test.mdl', 'wb') as outfile:
            pickle.dump(new_nn, outfile)
    else:
        with open('test.mdl', 'rb') as infile:
            new_nn = pickle.load(infile)

    #TODO: Do we want split to be mutating or do we copy it?
    new_nn.color_network()

    results = []
    for i in range(c.n):
        diff_net = new_nn.create_difference_network(i)
        torch_diff = TorchAPI.nn2torch(diff_net)
        tf.saved_model.save(torch_diff, 'marabou_model/')
        # Check if INSIGNIFICANT
        if not MarabouAPI.verify_model('marabou_model/', 500):
            results.append(i)

    print("\n\n\nResults:")
    print("Significant:      \t", c.significant_vars)
    print("Found Significant:\t", results)

do(True)
