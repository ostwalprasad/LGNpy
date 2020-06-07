import pandas as pd
import numpy as np
import networkx as nx
import numbers
import math
import copy
from .Graph import Graph
import warnings
from .logging_config import Logger

log = Logger()


class LinearGaussian(Graph):
    def __init__(self):
        super().__init__()

    def __get_node_values(self, node):
        """
    Get mean and variance of node using Linear Gaussian CPD. Calcluated using finding betas
    """
        index_to_keep = [self.nodes.index(node)]
        index_to_reduce = [self.nodes.index(idx) for idx in list(self.g.pred[node])]
        values = self.__get_parent_calculated_means(list(self.g.pred[node]))
        val = {n: round(v, 3) for n, v in zip(list(self.g.pred[node]), values)}

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

        node_values = {n: round(v, 3) for n, v in zip(list(self.g.pred[node]), values)}
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
        self.inf_summary.loc[:, "Variance"] = list(np.around(np.diag(self.cov),3))

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
           # .iloc[np.lexsort((self.inf_summary.index, self.inf_summary['Evidence'].values))]
        )

        return self.inf_summary

    def run_inference(self, debug=True, return_results=True):
        """
        Run Inference on network with given evidences.
        """

        _log = log.setup_logger(debug=debug)
        _log.debug("Started")

        self.parameters = dict.fromkeys(self.nodes)
        if all(x == None for x in self.evidences.values()):
            _log.debug("No evidence was set. Proceeding without evidence")
        root_nodes = [
            x
            for x in self.g.nodes()
            if self.g.out_degree(x) >= 1 and self.g.in_degree(x) == 0
        ]
        print("initial root nods")
        print(root_nodes)
        self.calculated_means = copy.deepcopy(self.evidences)
        self.calculated_vars = dict.fromkeys(self.nodes)
        self.done_flags = dict.fromkeys(self.nodes)

        for node in root_nodes:
            self.done_flags[node] = True

        while not all(x == True for x in self.done_flags.values()):
            next_root_nodes = copy.deepcopy(root_nodes)
            root_nodes = []
            for node in next_root_nodes:
                _log.debug(
                    f"Calculating children of {node} : {list(self.g.succ[node].keys())}"
                )
                if self.calculated_means[node] == None:
                    self.calculated_means[node] = self.mean[self.nodes.index(node)]
                    _log.debug(
                        f"Evidence was not available for node {node}, so took mean."
                    )
                for child in self.g.succ[node]:
                    if self.done_flags[child] != True:
                        (
                            self.calculated_means[child],
                            self.calculated_vars[child],
                        ) = self.__get_node_values(child)
                        _log.debug(f"\tcalculated for {child}")
                        self.done_flags[child] = True
                        root_nodes.append(child)
                    else:
                        _log.debug(f"\t{child} already calculated")

        return self.__build_results()

    def __get_pure_root_nodes(self, graph):
        root_nodes = [
            x
            for x in graph.nodes()
            if graph.out_degree(x) >= 1 and graph.in_degree(x) == 0
        ]

        children_of_root_nodes = []
        for node in root_nodes:
            children_of_root_nodes.extend(list(graph.succ[node]))

        pure_children = []
        for node in children_of_root_nodes:
            node_parents = list(graph.pred[node])
            flag = False
            for parent in node_parents:
                if graph.in_degree(parent) != 0:
                    flag = True
            if not flag:
                pure_children.append(node)
        return list(set(pure_children))

    def __remove_pred_edges(self, node, graph):
        preds = graph.pred[node]
        for parent in list(preds):
            graph.remove_edge(parent, node)

    def __print_message(self,log_instance,node):
        log_instance.debug(f"Calculated:'{node}'= {round(self.calculated_means[node], 3)}")
        log_instance.debug(f"Parent nodes used: {self.parameters[node]['node_values']}")
        log_instance.debug(f"Beta calculated: {self.parameters[node]['node_betas']}")


    def run_inference2(self, debug=True, return_results=True):
        """
        Run Inference 2 on network with given evidences.
        """
        g_temp = copy.deepcopy(self.g)
        _log = log.setup_logger(debug=debug)
        _log.debug("Started")

        if all(x == None for x in self.evidences.values()):
            _log.debug("No evidences were set. Proceeding without evidence")

        self.parameters = dict.fromkeys(self.nodes)
        self.calculated_means = copy.deepcopy(self.evidences)
        self.calculated_vars = dict.fromkeys(self.nodes)
        self.done_flags = dict.fromkeys(self.nodes)
        it=0
        while not nx.is_empty(g_temp):
            it+=1
            _log.debug(f"\n{'__'*10}\nIteration {it}")
            # self.draw_network("ssdf", g_temp)
            pure_children = self.__get_pure_root_nodes(g_temp)
            for child in pure_children:
                if self.evidences[child] is None:
                    self.calculated_means[child], self.calculated_vars[child] = self.__get_node_values(child)
                    self.__print_message(_log,child)
                else:
                    _log.debug(f"Skipped Calculating:'{child}' as evidence is available.")
                g_temp.remove_nodes_from(list(g_temp.pred[child]))

        # i=0
        # while not nx.is_empty(g2):
        #     i+=1
        #     if i >1:
        #         break
        #     #self.draw_network("ssdf", g2)
        #     pure_root_nodes = self.__get_pure_root_nodes(g2)
        #     print(f"Pure root nodes {pure_root_nodes}")
        #     for node in pure_root_nodes:
        #         self.done_flags[node] = True
        #
        #     for node in pure_root_nodes:
        #         if not g2.has_node(node):
        #             continue
        #         if self.calculated_means[node] is None:
        #             self.calculated_means[node] = self.mean[self.nodes.index(node)]
        #             _log.debug(f"Evidence was not available for node {node}, so took mean.")
        #         children = list(g2.succ[node])
        #         for child in children:
        #             if self.done_flags[child] != True:
        #                 self.calculated_means[child],self.calculated_vars[child],= self.__get_node_values(child)
        #                 _log.debug(f"calculated {child}', using parents {list(g2.pred[child])}")
        #                 self.done_flags[child] = True
        #             self.__remove_pred_edges(child,g2)
        #         g2.remove_nodes_from(list(nx.isolates(g2)))
        return self.__build_results()

    def get_inference_results(self):
        return self.inf_summary
