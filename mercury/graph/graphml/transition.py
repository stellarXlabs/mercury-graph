import warnings

import pandas as pd
import networkx as nx

from numpy.linalg import matrix_power


class Transition:
    """
    Create an interface class to manage the adjacency matrix of a directed graph as a transition matrix. 
    This enables computing distributions of probabilities over the nodes after a given number of iterations.

    Args
        G (Graph): A `mercury.graph` graph. This graph must be created using G.auto_init()

    """

    def __init__(self):
        self.G = nx.DiGraph()


    def set_row(self, row_id, row_content):
        """
        Defines an entire row of the transition matrix in just one call. All previous values in the row will be cleared even if not
        defined in the call.

        Args:
            row_id (str): The node id of the row that will be cleared and defined. It will automatically add new nodes if necessary.
            row_content (dict): A dictionary of destination node ids and weights. The weights must be numeric and positive!

        Returns:
            (None): None (or raises an error)

        """
        old_keys = set(self.G.nodes)
        new_keys = [nk for nk in [row_id] + list(row_content.keys()) if nk not in old_keys]

        for key in new_keys:
            self.G.add_node(key)

        for dest in list(self.G[row_id].keys()):
            self.G.remove_edge(row_id, dest)

        for dest in row_content.keys():
            self.G.add_edge(row_id, dest, weight=float(row_content[dest]))


    def fit(self):
        """
        Converts the adjacency matrix into a transition matrix. Transition matrices are used to compute the distribution of probability
        of being in each of the nodes (or states) of a directed graph (or Markov process). The distribution for state s is:


        * $s_t = T*s_{t-1}$

        Where:

        T is the transition matrix. After calling.fit(), the adjacency matrix is the transition matrix. You can use .topandas() to see it.
        $s_{t-1}$ is the previous state.

        Note:
            If created using NetworkX directly, the name of the weight must be 'weight' and must be positive. The recommended way
            to create the graph is using .set_row() which will always name the weight as 'weight' but does not check the value.

        Returns:
            self (object): Fitted self (or raises an error)

        What .fit() does is scaling the non-zero rows to make them sum 1 as they are probability distributions and make the zero rows
        recurrent states. A recurrent state is a final state, a state whose next state is itself.

        """
        names = list(self.G.nodes)
        adj_m = nx.adjacency_matrix(self.G, weight="weight")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for i in range(adj_m.shape[0]):
                row = adj_m[[i], :]
                tot = row.sum()

                if tot == 0:
                    row[0, i] = 1
                else:
                    row = row / tot

                adj_m[[i], :] = row

        df = pd.DataFrame(adj_m.todense(), index=names, columns=names)
        self.G = nx.from_pandas_adjacency(df, create_using=nx.DiGraph)


    def to_pandas(self, num_iterations=1):
        """
        Returns the adjacency (which is the transition matrix after .fit() was called) for a given number of iterations as a pandas
        dataframe with labeled rows and columns.

        Args:
            num_iterations (int): If you want to compute the matrix for a different number of iterations, k, you can use this argument to
                raise the matrix to any non negative integer, since:

        * $s_{t+k} = T^k*s_t$

        Returns:
            (pd.DataFrame): The transition matrix for num_iterations.

        Note:
            This method does not automatically call .fit(). This allows inspecting the adjacency matrix as a pandas dataframe.
            The result of computing num_iterations will not make sense if .fit() has not been called before .topandas().

        """
        names = list(self.G.nodes)
        adj_m = nx.adjacency_matrix(self.G, weight="weight").todense()

        if num_iterations != 1:
            adj_m = matrix_power(adj_m, num_iterations)

        return pd.DataFrame(adj_m, index=names, columns=names)
