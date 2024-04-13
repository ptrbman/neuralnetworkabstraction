#!/usr/bin/env python3
from nna.torch_api import TorchAPI
import tensorflow as tf
from nna.marabou_api import MarabouAPI
from nna.torch_api import TorchAPI
import numpy as np

class Utils:
    """Contains static functions search for bounds in a difference network.
    """


    # If upper_bound is too low, false result is reported
    def binary_bounds(nn, upper_bound):
        """Finds an upper bound (lower than upper_bound) for a nn using binary
        search. Note: if the upper bound is greater than upper_bound, this method will
        return a false result!

        :param nn: network to find upper bound for
        :param upper_bound: upper_bound of search
        :returns: an upper bound for nn

        """
        results = []
        for i in range(nn.layers[0].inputs):
            diff_net = nn.create_difference_network(i)
            torch_diff = TorchAPI.nn2torch(diff_net)
            tf.saved_model.save(torch_diff, 'marabou_model/')

            lb = 0
            ub = upper_bound
            bound = ub / 2
            # We keep searching until upper bound and lower bound is within distance of two
            while ub > lb + 2:
                # If bound is respected, it is a new upper bound, else it is a new lower bound
                if MarabouAPI.check_bound('marabou_model/', bound):
                    ub = bound
                else:
                    lb = bound
                bound = ((ub - lb) / 2) + lb
            results.append(bound)

        return results

    def iterative_bounds(nn, upper_bound):
        """Finds an upper bound (lower than upper_bound) for a nn using iterative
        search. If the upper bound is greater than upper_bound, this method will
        return None.

        :param nn: network to find upper bound for
        :param upper_bound: upper_bound of search
        :returns: an upper bound for nn or None if none exists smaller than upper_bound

        """
        results = []
        for i in range(nn.layers[0].inputs):
            diff_net = nn.create_difference_network(i)
            torch_diff = TorchAPI.nn2torch(diff_net)
            tf.saved_model.save(torch_diff, 'marabou_model/')

            b = 1
            found_bound = False
            while not found_bound and b <= upper_bound:
                if MarabouAPI.check_bound('marabou_model/', b):
                    found_bound = True
                else:
                    b += 1

            if found_bound:
                results.append(b)
            else:
                results.append(None)
        return results
