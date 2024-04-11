# neuralnetworkabstraction

## Usage

### Installation

To install, begin by cloning repo.

```
   $ git clone https://github.com/ptrbman/neuralnetworkabstraction.git
```

Ensure that Python 3.x is installed and create a virtual environment:


```
   $ python -m venv nnenv
   $ source nnenv/bin/activate
   (nnenv) $
```

Then install prerequisites (TensorFlow, Torch, Marabou)

```
   (nnenv) $ pip install tensorflow torch maraboupy
```



### Quick Test
Do to a quick test of the framework, run the supplmented ``quick_test.py``:

```
   (nnenv) $ python quick_test.py
```

Note that since initial weights are non-deterministic, the quick test is not
guaranteed to succeed, but can be stuck in a local optimum. Furthermore, there is lots of text from the machine learning framework as well as the verification framework which should be irrelevant. The important part are the final three lines:

```
   ...
   Results:
   Significant:         [1, 3]
   Found Significant:   [1, 3]
```

Which shows that the inputs which are found significant by the method are indeed the significant ones.


Additional Test
---------------
For more thorough testing, we refer to the README-file in the experiments directory. It is also possible to use the API found in the [documentation](https://ptrbman.github.io/neuralnetworkabstraction/)
