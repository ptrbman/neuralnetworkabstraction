#!/usr/bin/env python3

import random

class Configuration():
    """Represents a configuration for data-generation.

    """


    def __init__(self, number_of_variables, lower_bound, upper_bound):
        """Configuration constructor

        :param number_of_variables:
        :param lower_bound: lowest coefficient
        :param upper_bound: highest coefficient
        :returns: A configuration with the supplied parameters

        """
        self.n = number_of_variables
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.coefficients = [random.randint(lower_bound, upper_bound) for _ in range(self.n)]


    def __str__(self):
        """ String representation of configuration

        :returns: string representation

        """
        lines = []
        lines.append("No of Vars: " + str(self.n))
        lines.append("Bounds: " + str(self.lower_bound) + "/" + str(self.upper_bound))
        lines.append("Coefficients: " + str(self.coefficients))
        return '\n'.join(lines)
