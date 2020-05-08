import pandas as pd
import numpy as np
import networkx as nx
import numbers
import math
import warnings
from .logging_config import Logger

log = Logger()

class LinearGaussian():
  def __init__(self):
    """
    Create NetworkX graph object at initialization
    """
    self.g = nx.DiGraph()

  def set_data(self,dataframe):
    """
    Set Data using Pandas DataFrame. Parameters are set automatically from DataFrame.
    No need to call set_parameters if data is set using this function.
    """
    if not isinstance(dataframe, pd.DataFrame):
      raise TypeError("Argument invalid. Please provide Pandas DataFrame")
    if len(dataframe.columns) <=1:
      raise ValueError(f"Dataframe contains only {dataframe.columns}")
    if not set(list((self.g.nodes))).issubset(list(dataframe.columns)):
      raise ValueError(f"DataFrame does not contain {np.setdiff1d(list(self.g.nodes),list(dataframe.columns))}")
    dataframe = dataframe[list(self.g.nodes)]
    self.data = dataframe.reindex(sorted(dataframe.columns), axis=1)
    self.nodes = list((self.data.columns))
    self.mean  = np.array(self.get_mean())
    self.cov = np.array(self.get_covariance())
    self.evidences = dict.fromkeys(self.nodes)

  def set_parameters(self,mean,cov): #TODO self.data.mean giving errors
    """
    Set Mean and covariance manually wihout data.
    """
    self.mean = np.array(mean)
    self.cov = np.array(cov)
    self.nodes = list(self.g.nodes)
    if cov.shape[0] != cov.shape[1]:
      raise ValueError("Covariance Matrix is not a square.")
    if not self.mean.shape[0] == self.cov.shape[0]:
      raise ValueError(f"Mean and covariance matrix does not have matching dimentions {mean.shape},{cov.shape}")
    if len(self.g.nodes) != self.mean.shape[0]:
      raise ValueError("Length of mean vector!= length of nodes")
    self.evidences = dict.fromkeys(self.nodes)

  def get_covariance(self):
    """
    Get covariance of data
    """
    return self.data.cov()

  def get_precision_matrix(self):
    """
    Returns Inverse Covariance matrix (or Precision or Information Matrix)
    """
    return np.linalg.inv(self.cov)

  def get_mean(self):
    """
    Get mean of data
    """
    return self.data.mean()

  def set_edge(self,u,v):
    """
    Set edge from u to v
    """
    if u == v:
      raise ValueError("Self Loops not allowed.")
    self.g.add_edge(u,v)

  def set_edges_from(self,edges):
    """
    Set edges of network using list of edge tuples
    """
    for edge in edges:
      if edge[0] == edge[1]:
        raise ValueError("Self loops not allowed")
      self.g.add_edge(edge[0],edge[1])

  def draw_network(self,filename,open=True):
    """
    Plot network using matplolib library
    """



    nx.drawing.nx_pydot.to_pydot(self.g).write_png(filename+'.png')      
    if open == True:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        import matplotlib as mpl
        mpl.rcParams['figure.dpi']= 200
        plt.figure(figsize = (10,20))
        img=mpimg.imread(filename+'.png')
        imgplot = plt.imshow(img)
        plt.show()

  def get_parents(self,node):
    """
    Get parents of node
    """
    return list(self.g.pred[node])

  def get_children(self,node):
    """
    Get children of node
    """
    return list(self.g.succ[node])

  def get_network_object(self):
    return self.g

  def network_summary(self): 
    """
    Summary of each nodes in network.+
    """
    summary_cols= ['Node','Mean','Std','Parents','Children']
    summary = pd.DataFrame(columns=summary_cols)
    for node in self.nodes:
        row =[node,
              round(self.data[node].mean(),4),
              round(self.data[node].std(),4),
              self.get_parents(node),
              self.get_children(node)]
        summary.loc[len(summary)] = row
    return summary

  def get_network_info(self):
    pass
    log.info(f"Total Nodes: {len(self.nodes)}")

  def plot_distributions(self,save=False,filename=None):
    """
    KDE Plot of all the variables along with mean and standard deviation
    """

    import seaborn as sns
    import matplotlib.pyplot as plt

    rows=math.ceil(len(self.data.columns)/4)
    fig, ax = plt.subplots(ncols=4,
                           nrows=rows,
                           figsize=(12, rows*2))
    fig.tight_layout()
    for idx,axis in enumerate(ax.flatten()):
            sns.distplot(self.data.iloc[:, idx].dropna(),
                        norm_hist=False,
                        ax=axis,
                        label="")
            axis.set_title(self.data.columns[idx]) 
            axis.set_xlabel('')
            plt.text(0.2, 
                    0.8,
                    f'u:{round(self.data.iloc[:, idx].mean(),2)}\nsd={round(self.data.iloc[:, idx].std(),2)}', 
                    ha='center', 
                    va='center', 
                    transform=axis.transAxes)
            if idx == len(self.data.columns)-1:
              break
    plt.subplots_adjust(hspace = 0.4)   
    if save == True:
        plt.savefig(filename+'.png')
    plt.show()


  
  def get_nodes(self):
    """
    Get list of nodes in network
    """
    return list(self.g.nodes)

  def get_edges(self):

    return list(self.g.edges)

  def get_node_values(self,node):
    """
    Get mean and variance of node using Linear Gaussian CPD. Calcluated using finding betas
    """
    index_to_keep = [self.nodes.index(node)]
    index_to_reduce =  [self.nodes.index(idx) for idx in list(self.g.pred[node])]
    values = self._get_parent_calculated_means(list(self.g.pred[node]))

    mu_j = self.mean[index_to_keep]
    mu_i = self.mean[index_to_reduce]

    sig_i_j = self.cov[np.ix_(index_to_reduce, index_to_keep)]
    sig_j_i = self.cov[np.ix_(index_to_keep, index_to_reduce)]
    sig_i_i_inv = np.linalg.inv(self.cov[np.ix_(index_to_reduce, index_to_reduce)])
    sig_j_j = self.cov[np.ix_(index_to_keep, index_to_keep)]

    covariance = sig_j_j - np.dot(np.dot(sig_j_i, sig_i_i_inv), sig_i_j)
    beta_0 = mu_j - np.dot(np.dot(sig_j_i, sig_i_i_inv),mu_i)
    beta = np.dot(sig_j_i,sig_i_i_inv)

    new_mu = beta_0 + np.dot(beta,values)
    self.parameters[node]={'beta':list(beta_0)+list(beta[0])}
    
    return new_mu[0],covariance[0][0]

  def set_evidences(self,evidence_dict):
    """
    Set evidence using dictionary key,value pairs
    """
    if not isinstance(evidence_dict,dict):
      raise ValueError("Please provide dictionary")

    for key,val in evidence_dict.items():
      if key not in self.nodes:
        raise ValueError(f"'{key}'' node is not available in network")
      if not isinstance(val,numbers.Number):
        raise ValueError(f"Node '{key}'s given evidence is not a number. It's ({(val)})'")
      self.evidences[key] =val

  def get_evidences(self):
    """
    Get evidences if they are set
    """
    return self.evidences

  def _get_parent_calculated_means(self,nodes):
    """
    Get evidences of parents given node name list
    """
    pa_e = []
    for node in nodes:
      ev = self.calculated_means[node]
      if ev is None:
        ev = self.mean[self.nodes.index(node)]
      pa_e.append(ev)
    return pa_e

  def clear_evidences(self):
    """
    Clear evidences 
    """
    self.evidences = dict.fromkeys(self.nodes)
    pass

  def get_model_parameters(self):
    """
    Get parameters for each node
    """
    return self.parameters
    pass

  def run_inference(self,debug=True):
    """
    Run Inference on network with given evidences.
    """
    _log= log.setup_logger(debug=debug)
    print(_log)
    _log.debug("Started")
    self.parameters=dict.fromkeys(self.nodes)
    if all(x==None for x in self.evidences.values()):
      _log.debug("No evidence was set. Proceeding without evidence")
    leaf_nodes = [x for x in self.g.nodes() if self.g.out_degree(x)>=1 and self.g.in_degree(x)==0]
    self.calculated_means = self.evidences
    self.calculated_vars = dict.fromkeys(self.nodes)
    self.done_flags = dict.fromkeys(self.nodes)
    for node in leaf_nodes:
      self.done_flags[node]= True
    while not all(x==True for x in self.done_flags.values()):
      next_leaf_nodes = leaf_nodes
      leaf_nodes = []
      for node in next_leaf_nodes:
        _log.debug(f"Calculating children of {node} : {list(self.g.succ[node].keys())}")
        if self.calculated_means[node] == None:
          self.calculated_means[node] = self.mean[self.nodes.index(node)]
          _log.debug(f"Evidence wasn't available for node {node}, so took mean.")
        for child in self.g.succ[node]:
          if self.done_flags[child] != True:
            self.calculated_means[child],self.calculated_vars[child] = self.get_node_values(child)
            _log.debug(f"\tcalculated for {child}")
            self.done_flags[child]=True
            leaf_nodes.append(child)
          else:
            _log.debug(f"\t{child} already calculated")
    return self.calculated_means,self.calculated_vars

