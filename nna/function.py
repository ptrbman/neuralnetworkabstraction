#!/usr/bin/env python3

import random

class Function():
    """Represents a function for data-generation.

    """
    def __init__(self, number_of_variables, lower_bound, upper_bound):
        """Function constructor

        :param number_of_variables:
        :param lower_bound: lowest coefficient
        :param upper_bound: highest coefficient
        :returns: A function with the supplied parameters

        """
        self.n = number_of_variables
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound


class LinearFunction(Function):
    def __init__(self, number_of_variables, lower_bound, upper_bound):
        super().__init__(number_of_variables, lower_bound, upper_bound)
        self.coefficients = [random.randint(lower_bound, upper_bound) for _ in range(self.n)]

    def evaluate(self, xs):
        """Computes the function defined by config for inputs xs.

        :param xs: input values for the x-variables.
        :param config: config describing function.
        :returns: the resulting sum.

        """
        sum = 0
        for i in range(self.n):
            sum += self.coefficients[i]*xs[i]
        return sum



    def __str__(self):
        """ String representation of function

        :returns: string representation

        """
        return "Linear: " + str(self.coefficients) + " " + str(self.lower_bound) + " " + str(self.upper_bound)


class PairWiseFunction(Function):
    def __init__(self, number_of_variables, lower_bound, upper_bound):
        super().__init__(number_of_variables, lower_bound, upper_bound)
        self.coefficients = [random.randint(lower_bound, upper_bound) for _ in range((self.n*(self.n-1))//2)]

    def evaluate(self, xs):
        """Computes the function defined by config for inputs xs.

        :param xs: input values for the x-variables.
        :param config: config describing function.
        :returns: the resulting sum.

        """
        sum = 0
        c = 0
        for i in range(self.n):
            for j in range(i+1, self.n):
                sum += self.coefficients[c]*xs[i]*xs[j]
        return sum



    def __str__(self):
        """ String representation of function

        :returns: string representation

        """
        return "PairWise: " + str(self.coefficients) + " " + str(self.lower_bound) + " " + str(self.upper_bound)



class PairsFunction(Function):
    def __init__(self, number_of_variables, lower_bound, upper_bound):
        super().__init__(number_of_variables, lower_bound, upper_bound)
        self.coefficients = [random.randint(lower_bound, upper_bound) for _ in range(self.n//2)]

    def evaluate(self, xs):
        """Computes the function defined by config for inputs xs.

        :param xs: input values for the x-variables.
        :param config: config describing function.
        :returns: the resulting sum.

        """
        sum = 0
        for i in range(0, self.n, 2):
            sum += self.coefficients[i//2]*xs[i]*xs[i+1]
        return sum


    def __str__(self):
        """ String representation of function

        :returns: string representation

        """
        return "PairWise: " + str(self.coefficients) + " " + str(self.lower_bound) + " " + str(self.upper_bound)
