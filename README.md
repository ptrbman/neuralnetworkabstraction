# neuralnetworkabstraction

![NNA logo](https://raw.githubusercontent.com/ptrbman/neuralnetworkabstraction/master/docs/nna_logo.png)

This software demonstrates the approach of identifying insignificant inputs of a neural network using abstraction.

## Usage

### Installation

To install, begin by cloning repo.

```
   $ git clone https://github.com/ptrbman/neuralnetworkabstraction.git
```

Build the Docker image using


```
   docker build -t abstraction-test-environment .
```

Then start the container by using

```
   docker run -it -p 8888:8888 -v $(pwd):/workspace abstraction-test-environment
```



### Quick Test
Do to a quick test of the framework, run the supplmented ``quick_test.py``:

```
   $ python quick_test.py
```

Note that since initial weights are non-deterministic, the quick test is not
guaranteed to succeed, but can be stuck in a local optimum. Furthermore, there is lots of text from the machine learning framework as well as the verification framework which should be irrelevant. The important part are the final lines:

```
   ...
Coefficients:            [39, 18, 55, 100, 6, 29]
Input   Binary  Iterative:
0        38.28125        40
1        17.96875        19
2        55.46875        56
3        99.21875        None
4        7.03125         7
5        28.90625        30
```

Which shows that the coefficients randomly generated are correctly identified (of course, in every run the exact numbers will be different).


Additional Test
---------------
For more thorough testing, we refer to the README-file in the experiments directory. It is also possible to use the API found in the [documentation](https://ptrbman.github.io/neuralnetworkabstraction/)
