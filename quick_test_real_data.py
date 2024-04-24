#!/usr/bin/env python3

from nna.function import LinearFunction
from nna.model_generator import ModelGenerator
from nna.torch_api import TorchAPI
from nna.utils import Utils
import numpy as np

import pickle



def quick_test_real_data(csv_file,num_inputs,generate=False):
    nn = None
    upper_bound = 1

    if generate:
        torch_model = ModelGenerator.csv2model(csv_file,
            layer_sizes=[num_inputs, 10, 9, 1],epochs=20)
        nn = TorchAPI.torch2nn(torch_model)

        with open(f"test_real_{csv_file}.mdl", "wb+") as outfile:
            pickle.dump(nn, outfile)
    else:
        with open(f"test_real_{csv_file}.mdl", "rb") as infile:
            nn = pickle.load(infile)

    # TODO: we assume all inputs are 0 or 1
    # TODO: Do we want split to be mutating or do we copy it?
    nn.color_network()

    max_bound = 10
    binary_bounds = Utils.binary_bounds(nn, max_bound, upper_bound)
    # iterative_bounds = Utils.iterative_bounds(nn, max_bound, ub)

    ret = []
    ret.append("Real data from file:      \t" + str(csv_file))
    ret.append("Input\t  Binary:")
    for i, binary in enumerate(binary_bounds):
        ret.append(str(i) + "\t  " + str(binary))
    return "\n".join(ret)

all_results = []
CSV_13_INPUTS = "data/inputs_13_1_output.csv"
CSV_17_INPUTS = "data/inputs_17_1_output.csv"
CSV_27_INPUTS = "data/inputs_27_1_output.csv"
csvs_inputs = {CSV_13_INPUTS:13,CSV_17_INPUTS:17,CSV_27_INPUTS:27}

for csv,num in csvs_inputs.items():
    print (f"Accessing file {csv} with {num} inputs...")
    ret = quick_test_real_data(csv,num,generate=True)
    all_results.append(ret)
    
print("Done")
for r in all_results:
    print("............................")
    print(r)
