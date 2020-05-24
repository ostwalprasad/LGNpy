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

  def _get_conditionals(self):
      """
      """
      # Find out indexes to be conditioned based on evidences
      index_to_keep = [self.nodes.index(idx) for idx,val in self.evidences.items() if val is None ]
      index_to_remove = [self.nodes.index(idx) for idx, val in self.evidences.items() if val is not None]

      evidence_list = np.array([val for idx, val in self.evidences.items() if val is not None])

      # Find out modified Precision Matrix and H vector
      prec_j_i = self.precision_matrix[np.ix_(index_to_keep,index_to_remove)]
      self.hvectormod = self.hvector[index_to_keep] - prec_j_i.dot(evidence_list)
      self.precision_matrixmod = self.precision_matrix[np.ix_(index_to_keep, index_to_keep)]

  def _run_gabp(self,iterations,epsilon):

    """
    Run Message passing algorithm
    """
    j = np.diag(np.diag(self.precision_matrixmod))
    h = np.diag(self.hvectormod)
    p = self.precision_matrixmod
    self.iterations = iterations
    for n in range(self.iterations):
      for a in range(len(j)):
        for b in range(len(j)):
          if a != b and p[a][b]!= 0:
            j[a][b] = -p[a][b] * p[b][a] * (1/(sum(j[:,a]) - j[b][a]))
            h[a][b] = -p[a][b] * (1/(sum(j[:,a]) - j[b][a])) * (sum(h[:,a]) - h[b][a])
      if sum(sum(h - oldh)) < eps:
          print("converged")
    self.j=j
    self.h=h

  def _infer_marginals(self):
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


  def run_inference(self,iterations,debug=False):
        self._get_conditionals()
        self._run_gabp(iterations)



