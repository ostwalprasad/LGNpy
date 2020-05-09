class GaussianBP():
"""
Gaussian Belief Propogation (Message Passing algorithm) for Gaussian Graphical Models
"""

  def __init__(self):
    pass

  def set_parameters(self,precision_matrix,mean_vector,nodes):
    """
    Set precision matrix(j), mean vector(h) and node names
    """
    self.precision_matrix = precision_matrix
    self.mean_vector = mean_vector
    self.nodes = nodes

  def fit(self,iterations):
    """
    Run Message passing algorithm
    """
    j = np.diag(np.diag(self.precision_matrix))
    h = np.diag(self.mean_vector)
    self.iterations = iterations
    for n in range(self.iterations):
      for a in range(len(j)):
        for b in range(len(j)):
          if a != b and p[a][b]!= 0:
            j[a][b] = -p[a][b]*p[b][a]*(1/(sum(j[:,a]) - j[b][a]))
            h[a][b] = -p[a][b]*(1/(sum(j[:,a]) - j[b][a]))*(sum(h[:,a]) - h[b][a])
    self.j=j
    self.h=h

  def infer_marginals(self):
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

  def get_precision(self,cov):
    """
    Get precision matrix from covariance matrix
    """
    return np.linalg.inv(cov)
