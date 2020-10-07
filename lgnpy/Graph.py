import pandas as pd
import numpy as np
import networkx as nx
import numbers
import math
from unittest.mock import patch
from .logging_config import Logger


log = Logger()


class Graph:
    """
    Common class for graphs, evidences and EDA.
    """

    def __init__(self):
        """
        Create NetworkX graph object at initialization
        """
        self.g = nx.DiGraph()

    def set_data(self, dataframe):
        """Set Data using Pandas DataFrame. Parameters(mean,cov) are set automatically from DataFrame.

        Parameters
        ----------
        dataframe: Pandas Dataframe containing columns with all nodes.

        Returns
        -------
        None

        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Argument invalid. Please provide Pandas DataFrame")
        if len(dataframe.columns) <= 1:
            raise ValueError(f"Dataframe contains only {dataframe.columns}")
        if not set(list((self.g.nodes))).issubset(list(dataframe.columns)):
            raise ValueError(
                f"DataFrame does not contain {np.setdiff1d(list(self.g.nodes), list(dataframe.columns))}"
            )
        dataframe = dataframe[list(self.g.nodes)]
        self.data = dataframe.reindex(sorted(dataframe.columns), axis=1)
        self.nodes = list((self.data.columns))
        self.mean = np.array(self.get_mean())
        self.cov = np.array(self.get_covariance())
        self.precision_matrix = np.linalg.inv(self.cov)
        self.hvector = self.precision_matrix.dot(self.mean)
        self.evidences = dict.fromkeys(self.nodes)

    def set_parameters(self, mean, cov):  # TODO self.data.mean giving errors
        """Set Mean and covariance manually without pandas dataframe.

        NOT IMPLEMENTED YET.
        Parameters
        ----------
        mean: Mean Vector
        cov: Covariance Matrix

        Returns
        -------
        None

        """
        raise ValueError("Not implemented yet.")
        # self.mean = np.array(mean)
        # self.cov = np.array(cov)
        # self.nodes = list(self.g.nodes)
        # if cov.shape[0] != cov.shape[1]:
        #     raise ValueError("Covariance Matrix is not a square.")
        # if not self.mean.shape[0] == self.cov.shape[0]:
        #     raise ValueError(f"Mean and covariance matrix does not have matching dimentions {mean.shape},{cov.shape}")
        # if len(self.g.nodes) != self.mean.shape[0]:
        #     raise ValueError("Length of mean vector!= length of nodes")
        # self.evidences = dict.fromkeys(self.nodes)

    def get_covariance(self):
        """Returns  Covariance matrix

        Returns
        -------
        ndarray: nxn array of Covariance matrix

        """
        return self.data.cov()

    def get_precision_matrix(self):
        """Returns Inverse Covariance matrix (or Precision or Information Matrix)

        Returns
        -------
        ndarray: nxn array of precision matrix

        """
        return self.precision_matrix

    def get_mean(self):
        """Get mean of data

        Returns
        -------
        list of means

        """
        return self.data.mean()

    def set_edge(self, u, v):
        """Set single edge from u to v
        Parameters
        ----------
        u: From node
        v: To node

        Returns
        -------
        None

        """
        if u == v:
            raise ValueError("Self Loops not allowed.")
        self.g.add_edge(u, v)

    def set_edges_from(self, edges):
        """Set edges of network using list of edge tuples

        Parameters
        ----------
        edges: list of tuples of edges

        Returns
        -------
        None

        """
        for edge in edges:
            if edge[0] == edge[1]:
                raise ValueError("Self loops not allowed")
            self.g.add_edge(edge[0], edge[1])

    def get_parents(self, node):
        """

        Parameters
        ----------
        node: Node for which parents need to be returned

        Returns
        -------
        List of parent nodes

        """
        return list(self.g.pred[node])

    def get_children(self, node):
        """Get children of nodes

        Parameters
        ----------
        node: Node

        Returns
        -------
        list: list of children if any

        """

        return list(self.g.succ[node])

    def get_siblings(self, node):
        """Get sibling of node

        Parameters
        ----------
        node: Node

        Returns
        -------
        list of siblings if any

        """
        successors = list(self.g.succ[node])
        siblings = []
        for s in successors:
            siblings.extend(list(self.g.pred[s]))
        return list(set(siblings))

    def get_neighbors(self,node):
        """Get neighbors of node

        Parameters
        ----------
        node: Node

        Returns
        -------
        list of neighbors if any

        """
        return list(nx.all_neighbors(self.g,node))

    def get_nodes(self):
        """ Get list of nodes in network

        Returns
        -------
        list: List of nodes

        """
        return list(self.g.nodes)

    def get_edges(self):
        """Get list of edges in network

        Returns
        -------
        list: List of edges

        """

        return list(self.g.edges)

    def has_parents(self,node):
        """Check if node has parents.

        Parameters
        ----------
        node: Check if this node has parents

        Returns
        -------
        bool: True if has parents, False otherwise

        """
        parents = self.get_parents(node)
        return True if len(parents)!=0 else False

    def has_children(self,node):
        """Check if node has children.

               Parameters
               ----------
               node: Check if this node has children

               Returns
               -------
               bool: True if has children, False otherwise

               """
        parents = self.get_children(node)
        return True if len(parents)!=0 else False

    def remove_nodes(self, nodes):
        """Remove selected nodes from network

        Parameters
        ----------
        nodes: list of nodes

        Returns
        -------
        None

        """
        self.g.remove_nodes_from(nodes)


    def set_evidences(self, evidence_dict):
        """Set evidence using dictionary key,value pairs

        Parameters
        ----------
        evidence_dict: dictionary of evidence

        Returns
        -------
        None

        """
        if not isinstance(evidence_dict, dict):
            raise ValueError("Please provide dictionary")

        for key, val in evidence_dict.items():
            if key not in self.nodes:
                raise ValueError(f"'{key}'' node is not available in network")
            if not isinstance(val, numbers.Number):
                raise ValueError(
                    f"Node '{key}'s given evidence is not a number. It's ({val})'"
                )
            self.evidences[key] = val

    def get_evidences(self):
        """Get evidences if they are set

        Returns
        -------
        dict: Evidences with keys as nodes.
        """
        return self.evidences

    def clear_evidences(self):
        """Clear evidences

        """
        self.evidences = dict.fromkeys(self.nodes)

    def get_network_object(self):
        """
        Get NetworkX object

        Returns
        -------
        object: NetworkX instance of graph

        """
        return self.g
    
    def network_to_pandas(self):
        """
        Returns network in pandas format
        """
        return nx.to_pandas_dataframe(self.g)

    def network_summary(self):
        """
        Summary of each nodes in network.

        Returns
        -------
        Dataframe: Summary

        """
        summary_cols = ["Node", "Mean", "Std", "Parents", "Children"]
        summary = pd.DataFrame(columns=summary_cols)
        for node in self.nodes:
            row = [
                node,
                round(self.data[node].mean(), 4),
                round(self.data[node].std(), 4),
                self.get_parents(node),
                self.get_children(node),
            ]
            summary.loc[len(summary)] = row
        return summary

    def draw_network(self, filename, graph=None,correlation_annotation=True,open=True):
        """Draw Network using Graphviz and PyDot.

        This method used Graphviz library to draw graphs, so Graphviz needs to be installed.

        Download and install it from here : https://graphviz.gitlab.io/download/
        You might need to add environment variable for Graphviz.

        Parameters
        ----------
        filename: Image will wil stored with this filename.
        graph: Graph instance to plot. Not needed if plotting default graph og object
        correlation_annotation: True if pearson correlation coefficient needs to be written on edges
        open: Open file after storing.

        Returns
        -------
        Doesn't return anything but plots network if open=True

        """

        if graph is None:
            graph=self.g

        for edge in list(graph.edges):
            if correlation_annotation:
                graph.edges[edge[0],edge[1]]['label'] = str(round(self.data[[edge[0],edge[1]]].corr().iloc[0,1],1))
            graph.edges[edge[0], edge[1]]['fontname'] = 'Arial'
            graph.edges[edge[0], edge[1]]['fontsize'] = 10

        for node in graph.nodes:
            graph.nodes[node]['fontname'] = 'Arial'

        nx.drawing.nx_pydot.to_pydot(graph).write_png(filename + ".png")
        if open:
            # Import plotting libraries only if required
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg
            import matplotlib as mpl

            mpl.rcParams["figure.dpi"] = len(self.nodes)*15
            img = mpimg.imread(filename + ".png")
            plt.imshow(img)
            plt.show()

    def plot_distributions(self, save=False, filename=None):
        """KDE Plot of all the variables along with mean and standard deviation


        Parameters
        ----------
        save: Boolean, True saved image to directory
        filename: Name of File if saving.

        Returns
        -------
        Nothing. But plots diagram.

        """

        import seaborn as sns
        import matplotlib.pyplot as plt

        columns = 5
        sns.set(font_scale=1.0)
        rows = math.ceil(len(self.data.columns) / columns)
        fig, ax = plt.subplots(ncols=columns, nrows=rows, figsize=(12, rows * 2))

        fig.tight_layout()
        for idx, axis in enumerate(ax.flatten()):
            sns.distplot(
                self.data.iloc[:, idx].dropna(), norm_hist=False, ax=axis, label=""
            )

            axis.set_title(self.data.columns[idx])
            axis.set_xlabel("")

            axis.yaxis.set_major_formatter(plt.NullFormatter())
            plt.text(
                0.2,
                0.8,
                f"u:{round(self.data.iloc[:, idx].mean(), 2)}\nsd={round(self.data.iloc[:, idx].std(), 2)}",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            if idx == len(self.data.columns) - 1:
                break
        plt.subplots_adjust(hspace=0.4, wspace=0.1)
        if save:
            plt.savefig(filename + ".png")
        plt.show()
