import pandas as pd
import numpy as np
import networkx as nx
import numbers
import math
from .Graph import Graph
import warnings
from .logging_config import Logger

log = Logger()


class GaussianBP(Graph):
  """
  Gaussian Belief Propogation (Message Passing algorithm) for Gaussian Graphical Models
  """

  def __init__(self):
    super().__init__()
    pass

  def __fit(self,iterations):
    """
    Run Message passing algorithm
    """
    j = np.diag(np.diag(self.precision_matrix))
    h = np.diag(self.mean_vector)
    p = self.precision_matrix
    self.iterations = iterations
    for n in range(self.iterations):
      for a in range(len(j)):
        for b in range(len(j)):
          if a != b and p[a][b]!= 0:
            j[a][b] = -p[a][b]*p[b][a]*(1/(sum(j[:,a]) - j[b][a]))
            h[a][b] = -p[a][b]*(1/(sum(j[:,a]) - j[b][a]))*(sum(h[:,a]) - h[b][a])
    self.j=j
    self.h=h

  def __infer_marginals(self):
    """
    Infer marginal variance and mean for each node
    """
    self.infj=[]
    self.infh=[]
    for n in range(len(self.j)):
      self.infj.append(sum(self.j[:,n]))
      self.infh.append(sum(self.h[:,n]))
    self.infj = np.array(self.infj)
    self.infh = np.array(self.infh)
    return 1/self.infj,(1/self.infj)*self.infh


  def __build_precisions(self):
    """

    """
    self.modprecision = np.zeros()


  def run_inference(self,debug=False):
    pass

