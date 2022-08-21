import pandas as pd
import numpy as np
import copy
from .Graph import Graph
from .logging_config import Logger


class LinearGaussianExperimental(Graph):
    """
    Implements modified Linear Gaussian. Results not guaranteed. Experimental.
    """
    def __init__(self):
        super().__init__()
        self.log = Logger()

    def get_node_values(self, node, origin):
        """
        Get mean and variance of node using Linear Gaussian CPD.
        Calcluated using finding betas
        """
        neighbors = self.get_neighbors(node)
        if origin in neighbors:
            neighbors.remove(origin)

        index_to_keep = [self.nodes.index(node)]
        index_to_reduce = [self.nodes.index(idx) for idx in neighbors]
        values = self.__get_parent_calculated_means(neighbors)
        val = {n: round(v, 3) for n, v in zip(neighbors, values)}

        mu_j = self.mean[index_to_keep]
        mu_i = self.mean[index_to_reduce]

        sig_i_j = self.cov[np.ix_(index_to_reduce, index_to_keep)]
        sig_j_i = self.cov[np.ix_(index_to_keep, index_to_reduce)]
        sig_i_i_inv = np.linalg.inv(self.cov[np.ix_(index_to_reduce, index_to_reduce)])
        sig_j_j = self.cov[np.ix_(index_to_keep, index_to_keep)]

        covariance = sig_j_j - np.dot(np.dot(sig_j_i, sig_i_i_inv), sig_i_j)
        beta_0 = mu_j - np.dot(np.dot(sig_j_i, sig_i_i_inv), mu_i)
        beta = np.dot(sig_j_i, sig_i_i_inv)

        new_mu = beta_0 + np.dot(beta, values)

        node_values = {n: round(v, 3) for n, v in zip(neighbors, values)}
        node_beta = list(np.around(np.array(list(beta_0) + list(beta[0])), 2))
        self.parameters[node] = {"node_values": node_values, "node_betas": node_beta}
        return new_mu[0], covariance[0][0]

    def __get_parent_calculated_means(self, nodes):
        """
        Get evidences of parents given node name list
        """
        pa_e = []
        for node in nodes:
            ev = self.calculated_means[node]
            if ev is None:
                ev = self.mean[self.nodes.index(node)]
            else:
                pass
            pa_e.append(ev)
        return pa_e

    def get_model_parameters(self):
        """
        Get parameters for each node
        """
        return self.parameters

    def __build_results(self):
        """
        Make Pandas dataframe with the results.
        """

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
        self.inf_summary.loc[:, "Variance"] = list(np.around(np.diag(self.cov), 3))

        self.inf_summary["Mean_inferred"] = self.inf_summary.index.to_series().map(
            self.calculated_means
        )
        self.inf_summary["Variance_inferred"] = self.inf_summary.index.to_series().map(
            self.calculated_vars
        )
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

    def __print_message(self, log_instance, node):
        log_instance.debug(f"Calculated:'{node}'= {round(self.calculated_means[node], 3)}")
        log_instance.debug(f"Parent nodes used: {self.parameters[node]['node_values']}")
        log_instance.debug(f"Beta calculated: {self.parameters[node]['node_betas']}")

    def run_inference(self, inf_node, debug=True, return_results=True):
        """
        Run Inference on network with given evidences.
        """

        g_temp = copy.deepcopy(self.g)
        self._log = self.log.setup_logger(debug=debug)
        self._log.debug("Started")

        if all(x is None for x in self.evidences.values()):
            self._log.debug("No evidences were set. Proceeding without evidence")

        self.parameters = dict.fromkeys(self.nodes)
        self.calculated_means = copy.deepcopy(self.evidences)
        self.calculated_vars = dict.fromkeys(self.nodes)
        self.done_flags = dict.fromkeys(self.nodes)

        def recurse(current_node, origin):
            self._log.debug(f"Recursing for {current_node} with {origin}")
            for p in self.get_neighbors(current_node):
                if len(self.get_neighbors(p)) > 1:
                    if p != origin:
                        recurse(p, current_node)
                        self.calculated_means[p], self.calculated_vars[p] = self.get_node_values(p, current_node)
                        self.__print_message(self._log, p)

        recurse(inf_node, inf_node)
        self.calculated_means[inf_node], self.calculated_vars[inf_node] = self.get_node_values(inf_node, "")
        self.__print_message(self._log, inf_node)
        return self.__build_results()

    def get_inference_results(self):
        return self.inf_summary
