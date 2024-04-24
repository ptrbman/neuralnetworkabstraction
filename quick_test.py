#!/usr/bin/env python3

from nna.function import *
from nna.model_generator import ModelGenerator
from nna.torch_api import TorchAPI
from nna.utils import Utils
import numpy as np

import pickle

def check_all_inputs(nn, counts):
    def generate_bitstrings(n):
        return np.array(list([np.array([int(bit)*10 for bit in format(i, '0{}b'.format(n))]) for i in range(2**n)]))

    tests = generate_bitstrings(nn.layers[0].inputs)
    tests = tests[0:counts]
    torch = TorchAPI.nn2torch(nn)
    results = []
    for i in range(0, len(tests), 32):
        tmp = torch.predict(tests[i:i+32])
        for r in tmp:
            results.append(r)
    for t, r in zip(tests, results):
        print(t, "\t", r)


def quick_test(generate=False):
    # f = LinearFunction(6, 1, 20)
    ub = 1
    f = LinearFunction(10, 1, 10)

    nn = None

    if generate:
        torch_model = ModelGenerator.generate_model(f, [f.n, 5, 3, 1], ub, 20, 20000)
        nn = TorchAPI.torch2nn(torch_model)

        with open('test.mdl', 'wb') as outfile:
            pickle.dump(nn, outfile)
    else:
        with open('test.mdl', 'rb') as infile:
            nn = pickle.load(infile)


    model = TorchAPI.nn2torch(nn)
    # print("Function:      \t", f)
    # check_all_inputs(nn, 20)

    #TODO: we assume all inputs are 0 or 1
    #TODO: Do we want split to be mutating or do we copy it?
    nn.color_network()

    max_bound = 50
    binary_bounds = Utils.binary_bounds(nn, max_bound, ub)
    # iterative_bounds = Utils.iterative_bounds(nn, max_bound, ub)

    ret = []
    ret.append("Function:      \t" + str(f))
    ret.append("Input\t  Binary:")
    for i, binary in enumerate(binary_bounds):
        ret.append(str(i) +  "\t  " + str(binary))

    # ret.append("Input\t  Binary\t  Iterative:")
    # for i, (binary, iterative) in enumerate(zip(binary_bounds, iterative_bounds)):
        # ret.append(str(i) + "\t  " + str(binary) + "\t  " + str(iterative))

    return '\n'.join(ret)



all_results = []
for i in range(10):
    ret = quick_test(True)
    all_results.append(ret)


print("Done")
for r in all_results:
    print("............................")
    print(r)
