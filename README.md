
<img src="https://github.com/ostwalprasad/LGNpy/raw/master/docs/images/logo.png" width="250" > 
<hr /> 

[![Build Status](https://travis-ci.org/ostwalprasad/LGNpy.svg?branch=master)](https://travis-ci.org/ostwalprasad/LGNpy)
![PyPI - License](https://img.shields.io/pypi/l/lgnpy)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lgnpy)
[![Documentation Status](https://readthedocs.org/projects/lgnpy/badge/?version=latest)](https://lgnpy.readthedocs.io/en/latest/?badge=latest) 
[![codecov](https://codecov.io/gh/ostwalprasad/LGNpy/branch/master/graph/badge.svg)](https://codecov.io/gh/ostwalprasad/LGNpy) 
[![Downloads](https://pepy.tech/badge/lgnpy)](https://pepy.tech/project/lgnpy)


## Gaussian Bayesian Networks -Representation, Learning and Inference

LGNs are Bayesian Networks where all the nodes have continuous data. Gaussians are surprisingly good approximation for many real world continuous distributions. This package helps in modelling the network, learning parameters through data and running inference with evidence(s)

#### Models Implemented:
1. Linear Gaussian Network
2. Gaussian Belief Propagation (Also called as _Message Passing Algorithm_ or _GaBP Algorithm_)

## Installation
```bash
$ pip install lgnpy
```

or clone the repository.

```bash
$ pip install https://github.com/ostwalprasad/lgnpy
```



## Getting Started

Here are steps for Linear Gaussian Model. Gaussian Belief Propagation Model is also similar. 
#### 	1. Create Network

<img src="https://raw.githubusercontent.com/ostwalprasad/LGNpy/master/docs/images/network.png" width="200" >
    
```python
import pandas as pd
import numpy as np
from lgnpy import LinearGaussian

lg = LinearGaussian()
lg.set_edges_from([('A', 'D'), ('B', 'D'), ('D', 'E'), ('C', 'E')])
```

####	2 Create Data and assign to it to network.

​	Create synthetic data for network using pandas and bind network with the data. There's no need to separately calculate means and covariance matrix.

```python
np.random.seed(42)
n=100
data = pd.DataFrame(columns=['A','B','C','D','E'])
data['A'] = np.random.normal(5,2,n)
data['B'] = np.random.normal(10,2,n)
data['D'] = 2*data['A'] + 3*data['B'] + np.random.normal(0,2,n)
data['C'] = np.random.normal(-5,2,n)
data['E'] = 3*data['C'] + 3*data['D'] + np.random.normal(0,2,n)

lg.set_data(data)
```

####	3. Set Evidence(s)

 Evidence are optional and can be set before running inference.

```python
 lg.set_evidences({'A':5,'B':10})
```

####	4. Run Inference 

For each node, CPT (Conditional Probability Distribution) is defined as::<br/>

<img src="https://raw.githubusercontent.com/ostwalprasad/LGNpy/master/docs/images/cpd.png" width="210" ><br/>

where, its parameters  are calculated using conditional distribution of parent(s) and nodes: <br/>

<img src="https://raw.githubusercontent.com/ostwalprasad/LGNpy/master/docs/images/betas.png"  width="180" > <br/>

`run_inference()` returns inferred means and variances of each nodes.

   ```python
lg.run_inference(debug=False)
   ```

   

## Additional Functions:

```python
lg.plot_distributions(save=False)
```
<img src="https://github.com/ostwalprasad/LGNpy/raw/master/docs/images/distributions.png" width="800" > <br/>

```python
lg.network_summary()
```
<br/><img src="https://raw.githubusercontent.com/ostwalprasad/LGNpy/master/docs/images/summary.png"  width="300" > <br/>
```python
lg.draw_network(filename='sample_network',open=True)
```
<br/><img src="https://raw.githubusercontent.com/ostwalprasad/LGNpy/master/docs/images/drawn_network.png"  width="200"> <br/>

## Examples

Notebook: [Linear Gaussian Networks](https://github.com/ostwalprasad/LGNpy/blob/master/examples/lgnpy_example.ipynb)

Notebook: [Gaussian Belief Propagation]() 

## References:

##### Linear Gaussian Networks

1. [Probabilistic Graphical Models - Principles and Techniques ](https://mitpress.mit.edu/books/probabilistic-graphical-models), Daphne Koller, Chapter 7.2

2. [Gaussian Bayesian Networks](https://cedar.buffalo.edu/~srihari/CSE674/Chap7/7.2-GaussBNs.pdf), Sargur Srihari

#####  Gaussian Belief Propogation 

1. [Probabilistic Graphical Models - Principles and Techniques ](https://mitpress.mit.edu/books/probabilistic-graphical-models), Daphne Koller, Chapter 14.2.3
2. [Gaussian Belief Propagation: Theory and Aplication](https://arxiv.org/abs/0811.2518), Danny Bickson
## License

MIT License Copyright (c) 2020, Prasad Ostwal

