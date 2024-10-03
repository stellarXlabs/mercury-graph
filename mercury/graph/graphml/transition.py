import warnings

import pandas as pd
import networkx as nx

from numpy.linalg import matrix_power

from mercury.graph.core import Graph


class Transition:
    """
    Create an interface class to manage the adjacency matrix of a directed graph as a transition matrix. 
    This enables computing distributions of probabilities over the nodes after a given number of iterations.

    Args
        G_markov_ (Graph): A `mercury.graph` Graph resulting from calling method fit() on a Graph, 
            where its adjacency matrix has been converted into a transition matrix.

    """

    def __init__(self):
        self.G_markov_ = None


    def fit(self, G: Graph):
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

        Args
            G (Graph): A `mercury.graph` Graph.

        Returns:
            self (object): Fitted self (or raises an error).

        What .fit() does is scaling the non-zero rows to make them sum 1 as they are probability distributions and make the zero rows
        recurrent states. A recurrent state is a final state, a state whose next state is itself.

        """
        names = list(G.networkx.nodes)
        adj_m = nx.adjacency_matrix(G.networkx, weight="weight", dtype=float)

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
        self.G_markov_ = Graph(nx.from_pandas_adjacency(df, create_using=nx.DiGraph))

        return self


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
            The result of computing num_iterations will not make sense if .fit() has not been called before .to_pandas().

        """
        if self.G_markov_ is None:
            raise ValueError("Error: fit() must be called first.")
        
        names = list(self.G_markov_.networkx.nodes)
        adj_m = nx.adjacency_matrix(self.G_markov_.networkx, weight="weight").todense()

        if num_iterations != 1:
            adj_m = matrix_power(adj_m, num_iterations)

        return pd.DataFrame(adj_m, index=names, columns=names)
