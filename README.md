# LGNpy

![Build Status](https://travis-ci.org/ostwalprasad/lgnpy.svg?branch=master) ![PyPI - License](https://img.shields.io/pypi/l/lgnpy) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lgnpy)

## Representation, Learning and Inference for Linear Gaussian Network



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

____

\begin{aligned}
P(A)=& \mathcal{N}\left(\mu_{A}, \sigma_{A}^{2}\right) \\
P(B)=& \mathcal{N}\left(\mu_{B}, \sigma_{B}^{2}\right) \\
P(C)=& \mathcal{N}\left(\mu_{C}, \sigma_{C}^{2}\right) \\
P(D | A, B)=\mathcal{N}\left(\tilde{\beta}_{D, 0}+\beta_{D, 1} A+\beta_{D, 2} B, \sigma_{D}^{2}\right) & \\
P(E | C, D)=\mathcal{N}\left(\beta_{E, 0}+\beta_{E, 1} C+\beta_{E, 2} D, \sigma_{E}^{2}\right)
\end{aligned}

### Getting Started

________

Here's an example on how to use LGNpy to 

```python
import pandas as pd
imoprt numpy as np
from lgnpy import LinearGaussian


```









