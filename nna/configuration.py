#!/usr/bin/env python3

class Configuration():
    """Represents a configuration for data-generation.

    """


    def __init__(self, number_of_variables, significant_variables, large, small):
        """Configuration constructor

        :param number_of_varibles: Total number of variables
        :param significant_variables: Indices of significant variables
        :param large: Coefficient of significant variables
        :param small: Coefficient of insignificant variables
        :returns: A configuration with the supplied parameters

        """

        self.n = number_of_variables
        self.significant_vars = significant_variables
        self.large = large
        self.small = small


    def __str__(self):
        """ String representation of configuration

        :returns: string representation

        """
        lines = []
        lines.append("No of Vars: " + str(self.n))
        lines.append("Significant: " + str(self.significant_vars))
        lines.append("Large/Small: " + str(self.large) + "/" + str(self.small))
        return '\n'.join(lines)
