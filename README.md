# LGNpy

![Build Status](https://travis-ci.org/ostwalprasad/lgnpy.svg?branch=master) ![PyPI - License](https://img.shields.io/pypi/l/lgnpy) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lgnpy)

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



### Linear Gaussian Networks

______________

**Example:**

 All the variables are Gaussian and all CPDs are Linear Gaussian. 

<a href="url"><img src="docs/images/sample_network.png" width="500" ></a>



**Model Parameters:**

The conditional density P(Y|X) is given by

<img src="docs/images/cpd.png" align="left" width="200" >



where,



<img src="docs/images/betas.png" align="left" width="180" >

jkkljkl
$$
x
$$

$$
\sqrt{2}
$$
  h<sub>&theta;</sub>(x) = &theta;<sub>o</sub> x + &theta;<sub>1</sub>x

### Getting Started

________

Here's an example on how to use LGNpy to 

```python
import pandas as pd
imoprt numpy as np
from lgnpy import LinearGaussian


```









