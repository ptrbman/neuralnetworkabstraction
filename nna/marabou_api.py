#!/usr/bin/env python3

from maraboupy import Marabou
from maraboupy import MarabouCore

class MarabouAPI:
    """Contains static functions to interface with Marabou verifier.
    """

    def check_bound(infile, bound, MAX_IN):
        """Verify if model found in infile is bounded above by bound.

        :param infile: file to load pytorch model
        :param bound: upper bound to verify
        :returns: True if bound is respected, otherwise False

        """
        network = Marabou.read_tf(infile, modelType="savedModel_v2")
        inputVars = network.inputVars[0][0]
        outputVar = network.outputVars[0][0]

        for var in inputVars:
            network.setLowerBound(var, 0)
            network.setUpperBound(var, MAX_IN)
        network.addInequality(outputVar, [-1.0], -bound)

        result, values, stats = network.solve()

        if result == "unsat":
            return True
        else:
            return False
