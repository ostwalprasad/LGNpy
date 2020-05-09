# LGNpy

![Build Status](https://travis-ci.org/ostwalprasad/lgnpy.svg?branch=master) ![PyPI - License](https://img.shields.io/pypi/l/lgnpy) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lgnpy) [![Documentation Status](https://readthedocs.org/projects/lgnpy/badge/?version=latest)](https://lgnpy.readthedocs.io/en/latest/?badge=latest)

## Representation, Learning and Inference for Linear Gaussian Networks

Features-
1. Network Representation
2. Parameter Learning through Pandas dataframe
3. Network related EDA
4. Inference with Evidence

### Installation

_______

```bash
$ pip install lgnpy
```

or clone the repository.

```bash
$ pip install https://github.com/ostwalprasad/lgnpy
```



______________

**References:**

[Probabilistic Graphical Models - Principles and Techniques ](https://mitpress.mit.edu/books/probabilistic-graphical-models), Daphne Koller, Chapter 7.2

[Gaussian Bayesian Networks](https://cedar.buffalo.edu/~srihari/CSE674/Chap7/7.2-GaussBNs.pdf), Sargur Srihari



**Example:**

 All the variables are Jointly Gaussian 

**Model Parameters:**

`run_inference()` function calculates following parameters for each node except root and evidences nodes.

<img src="docs/images/betas.png" align="left" width="180" >



### Getting Started

________

1. Let's create a simple network.

<img src="docs/images/network.png" width="200" >

```python
import pandas as pd
imoprt numpy as np
from lgnpy import LinearGaussian

lg = LinearGaussian()
lg.set_edges_from([('A', 'D'), ('B', 'D'), ('D', 'E'), ('C', 'E')])
```

2. Prepare synthetic data and bind it to the network.

   After data bind to network, it calculates mean and variance from network and assigns to individual nodes.

```python
np.random.seed(42)
n=100
data = pd.DataFrame(columns=['A','B','C','D','E'])
data['A'] = np.random.normal(0,2,n)
data['B'] = np.random.normal(5,3,n)
data['D'] = 2*data['A'] + 3*data['B'] + np.random.normal(0,2,n)
data['C'] = np.random.normal(2,2,n)
data['E'] = 3*data['C'] + 3*data['D'] + np.random.normal(0,2,n)

lg.set_data(data)
```

3. Set Evidence 

   ```
   lg.set_evidences({'A':7,'B':2})
   ```

4. Run Inference which returns inferred values of mean and variance using :

   <img src="docs/images/cpd.png" align="left" width="180" >

   where, model parameters are:

   <img src="docs/images/betas.png" align="left" width="180" >

   ```
   lg.run_inference(debug=False)
   
   >>>  ({'A': -0.20769303478818774,
   >>>    'B': 5.066913761149772,
   >>>    'C': 2.213680241394609,
   >>>    'D': 14.91514772007384,
   >>>    'E': 51.27447494338465},
   >>>   {'A': None,
   >>>    'B': None,
   >>>    'C': None,
   >>>    'D': 4.530868459203305,
   >>>    'E': 4.255827816965507})
   ```

   



