import pandas as pd
import numpy as np
import networkx as nx
import numbers
import math
from .Graph import Graph
import matplotlib.pyplot as plt
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
      Find out indexes to be conditioned based on evidences
      """
        self.index_to_keep = [
            self.nodes.index(idx) for idx, val in self.evidences.items() if val is None
        ]
        self.index_to_remove = [
            self.nodes.index(idx)
            for idx, val in self.evidences.items()
            if val is not None
        ]
        evidence_list = np.array(
            [val for idx, val in self.evidences.items() if val is not None]
        )

        # Find out modified Precision Matrix and H vector
        prec_j_i = self.precision_matrix[
            np.ix_(self.index_to_keep, self.index_to_remove)
        ]
        self.hvectormod = self.hvector[self.index_to_keep] - prec_j_i.dot(evidence_list)
        self.precision_matrixmod = self.precision_matrix[
            np.ix_(self.index_to_keep, self.index_to_keep)
        ]

    def _run_gabp(self, iterations, epsilon):

        """
    Run Message passing algorithm
    """
        self.broken=False
        j = np.diag(np.diag(self.precision_matrixmod))
        h = np.diag(self.hvectormod)
        p = self.precision_matrixmod
        self.iterations = iterations
        self.errors = []
        for n in range(self.iterations):
            oldh = h.copy()
            for a in range(len(j)):
                for b in range(len(j)):
                    if a != b and p[a][b] != 0:
                        j[a][b] = -p[a][b] * p[b][a] * (1 / (sum(j[:, a]) - j[b][a]))
                        h[a][b] = (
                            -p[a][b]
                            * (1 / (sum(j[:, a]) - j[b][a]))
                            * (sum(h[:, a]) - h[b][a])
                        )
            self.errors.append(sum(sum(h - oldh)))
            if abs(sum(sum(h - oldh))) == 0 or abs(sum(sum(h - oldh))) < epsilon:
                print(f"Converged in {n} iterations.")
                self.broken=True
                break
        if not self.broken:
            print(f"Did not converge in given {self.iterations}. You can use plot_errors() method to see convergence plot.")
            return False
        self.j = j
        self.h = h
        return True

    def _infer_marginals(self):
        """
    Infer marginal variance and mean for each node
    """
        self.infj = []
        self.infh = []
        for n in range(len(self.j)):
            self.infj.append(sum(self.j[:, n]))
            self.infh.append(sum(self.h[:, n]))
        self.infj = np.array(self.infj)
        self.infh = np.array(self.infh)
        return {"mean": (1 / self.infj) * self.infh, "var": 1 / self.infj}

    def __build_precisions(self):
        """
    """
        self.modprecision = np.zeros()

    def plot_errors(self):
        plt.plot(self.errors)

    def run_inference(self, iterations, epsilon, debug=False):
        self.inf_summary = pd.DataFrame(
            index=self.nodes,
            columns=[
                "Evidence",
                "Mean",
                "Mean_inferred",
                "Variance",
                "Variance_inferred",
            ],
        )
        self.inf_summary.loc[:, "Mean"] = self.mean
        self.inf_summary["Evidence"] = self.inf_summary.index.to_series().map(
            self.evidences
        )
        self.inf_summary.loc[:, "Variance"] = np.diag(self.cov)

        self._get_conditionals()
        if not self._run_gabp(iterations, epsilon):
            return self.inf_summary
        results = self._infer_marginals()

        self.inf_summary.loc[
            self.inf_summary.index[self.index_to_keep], "Mean_inferred"
        ] = results["mean"]
        self.inf_summary.loc[
            self.inf_summary.index[self.index_to_keep], "Variance_inferred"
        ] = results["var"]
        self.inf_summary["u_%change"] = (
            (self.inf_summary["Mean_inferred"] - self.inf_summary["Mean"])
            / self.inf_summary["Mean"]
        ) * 100
        self.inf_summary = (
            self.inf_summary.round(4)
            .replace(pd.np.nan, "", regex=True)
            .replace(0, "", regex=True)
        )
        return self.inf_summary
