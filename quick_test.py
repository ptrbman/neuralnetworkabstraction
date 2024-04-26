#!/usr/bin/env python3

from nna.function import LinearFunction
from nna.model_generator import ModelGenerator
from nna.torch_api import TorchAPI
from nna.utils import Utils
import numpy as np

import pickle


def check_all_inputs(nn, counts):
    def generate_bitstrings(n):
        return np.array(
            list(
                [
                    np.array([int(bit) * 10 for bit in format(i, "0{}b".format(n))])
                    for i in range(2**n)
                ]
            )
        )

    tests = generate_bitstrings(nn.layers[0].inputs)
    tests = tests[0:counts]
    torch = TorchAPI.nn2torch(nn)
    results = []
    for i in range(0, len(tests), 32):
        tmp = torch.predict(tests[i : i + 32])
        for r in tmp:
            results.append(r)
    for t, r in zip(tests, results):
        print(t, "\t", r)


def quick_test(generate=False):
    # f = LinearFunction(6, 1, 20)
    upper_bound = 1
    f = LinearFunction(10, 1, 10)

    nn = None

    if generate:
        torch_model = ModelGenerator.generate_model(
            function=f,
            layer_sizes=[f.n, 5, 3, 1],
            upper_bound=upper_bound,
            epochs=10,
            count=20000,
        )
        nn = TorchAPI.torch2nn(torch_model)

        with open("test.mdl", "wb") as outfile:
            pickle.dump(nn, outfile)
    else:
        with open("test.mdl", "rb") as infile:
            nn = pickle.load(infile)

    model = TorchAPI.nn2torch(nn)
    # print("Function:      \t", f)
    # check_all_inputs(nn, 20)

    # TODO: we assume all inputs are 0 or 1
    # TODO: Do we want split to be mutating or do we copy it?
    nn.color_network()

    max_bound = 50
    binary_bounds = Utils.binary_bounds(nn, max_bound, upper_bound)
    # iterative_bounds = Utils.iterative_bounds(nn, max_bound, ub)

    ret = []
    ret.append("Function:      \t" + str(f))
    ret.append("Input\t  Binary:")
    for i, binary in enumerate(binary_bounds):
        ret.append(str(i) + "\t  " + str(binary))

    # ret.append("Input\t  Binary\t  Iterative:")
    # for i, (binary, iterative) in enumerate(zip(binary_bounds, iterative_bounds)):
    # ret.append(str(i) + "\t  " + str(binary) + "\t  " + str(iterative))

    return "\n".join(ret)


def quick_test_real_data(csv_file,num_inputs,generate=False):
    nn = None
    upper_bound = 1

    if generate:
        torch_model = ModelGenerator.csv2model(csv_file,
            layer_sizes=[num_inputs, 10, 9, 1],)
        nn = TorchAPI.torch2nn(torch_model)

        with open(f"test_real_{csv_file}.mdl", "wb") as outfile:
            pickle.dump(nn, outfile)
    else:
        with open(f"test_real_{csv_file}.mdl", "rb") as infile:
            nn = pickle.load(infile)

    model = TorchAPI.nn2torch(nn)
    # print("Function:      \t", f)
    # check_all_inputs(nn, 20)

    # TODO: we assume all inputs are 0 or 1
    # TODO: Do we want split to be mutating or do we copy it?
    nn.color_network()

    max_bound = 50
    binary_bounds = Utils.binary_bounds(nn, max_bound, upper_bound)
    # iterative_bounds = Utils.iterative_bounds(nn, max_bound, ub)

    ret = []
    ret.append("Real data from file:      \t" + str(csv_file))
    ret.append("Input\t  Binary:")
    for i, binary in enumerate(binary_bounds):
        ret.append(str(i) + "\t  " + str(binary))
    return "\n".join(ret)

all_results = []
# for i in range(10):
#     ret = quick_test(True)
#     all_results.append(ret)

CSV_13_INPUTS = "data/inputs_13_1_output.csv"
CSV_17_INPUTS = "data/inputs_17_1_output.csv"
CSV_27_INPUTS = "data/inputs_27_1_output.csv"
csvs_inputs = {CSV_13_INPUTS:13,CSV_17_INPUTS:17,CSV_27_INPUTS:27}

for csv,num in csvs_inputs:
    ret = quick_test_real_data(csv,num,generate=True)
    all_results.append(ret)
    
print("Done")
for r in all_results:
    print("............................")
    print(r)
