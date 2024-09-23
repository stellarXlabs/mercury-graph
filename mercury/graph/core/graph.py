import pandas as pd
import networkx as nx

from mercury.graph.core.spark_interface import SparkInterface, pyspark_installed, graphframes_installed


class Graph:
    """ This is the main class in mercury.graph.

    This class seamlessly abstracts the underlying technology used to represent the graph. You can create a graph passing the following
    objects to the constructor:

    - A pandas DataFrame containing edges (with a keys dictionary to specify the columns)
    - A pyspark DataFrame containing edges (with a keys dictionary to specify the columns)
    - A networkx graph
    - A graphframes graph
    - A binary serialization of the object storing all its state.

    Bear in mind that the graph object is inmutable. This means that you can't modify the graph object once it has been created. If you
    want to modify it, you have to create a new graph object.

    The graph object provides:

    - Properties to access the graph in different formats (networkx, graphframes, dgl)
    - Properties with metrics and summary information that are calculated on demand and technology independent.
    - It is inherited by other graph classes in mercury-graph providing ML algorithms such as graph embedding, visualization, etc.
    """
    def __init__(self, data = None, keys = None, nodes = None):
        self._built = False
        self._as_networkx = None
        self._as_graphframe = None
        self._as_dgl = None
        self._degree = None
        self._closeness_centrality = None
        self._betweenness_centrality = None
        self._pagerank = None
        self._connected_components = None
        self._connected_components_c = None
        self._nodes_summary = None
        self._edges_summary = None

        self._number_of_nodes = 0
        self._number_of_edges = 0
        self._node_ix = 0

        if type(data) == pd.core.frame.DataFrame:
            self._from_pandas(data, nodes, keys)
            return

        if type(data) == nx.classes.graph.Graph:
            self._from_networkx(data)
            return

        spark_int = SparkInterface()

        if pyspark_installed and type(data) == spark_int.type_spark_dataframe:
            self._from_dataframe(data, nodes, keys)

        if graphframes_installed and type(data) == spark_int.type_graphframe:
            self._from_graphframes(data)

        self._from_binary(data)


    def __iter__(self):
        return self


    def __len__(self):
        return self._number_of_nodes


    def __str__(self):
        # TODO: This
        return '.'


    def __repr__(self):
        # TODO: This
        return '.'


    def __next__(self):
        if self._node_ix < 0 or self._node_ix >= self._number_of_nodes:
            self._node_ix = -1

        node_id = self._node_id(self._node_ix)
        self._node_ix += 1

        if self._node_ix == self._number_of_nodes:
            raise StopIteration

        return node_id


    def __del__(self):
        # TODO: This
        return


    def __getstate__(self):
        """ Used by pickle.dump() (See https://docs.python.org/3/library/pickle.html)
        """
        # TODO: This
        return '.'


    def __setstate__(self, state):
        """ Used by pickle.load() (See https://docs.python.org/3/library/pickle.html)
        """
        # TODO: This
        return


    def __deepcopy__(self, memo):
        # TODO: This
        return


    @property
    def networkx(self):
        if self._as_networkx is None:
            self._as_networkx = self._to_networkx()

        return self._as_networkx


    @property
    def graphframe(self):
        if self._as_graphframe is None:
            self._as_graphframe = self._to_graphframe()

        return self._as_graphframe


    @property
    def dgl(self):
        if self._as_dgl is None:
            self._as_dgl = self._to_dgl()

        return self._as_dgl


    @property
    def degree(self):
        if self.degree is None:
            self.degree = self._calculate_degree()
        return self.degree


    @property
    def closeness_centrality(self):
        if self._closeness_centrality is None:
            self._closeness_centrality = self._calculate_closeness_centrality()
        return self._closeness_centrality


    @property
    def betweenness_centrality(self):
        if self._betweenness_centrality is None:
            self._betweenness_centrality = self._calculate_betweenness_centrality()
        return self._betweenness_centrality


    @property
    def pagerank(self):
        if self._pagerank is None:
            self._pagerank = self._calculate_pagerank()
        return self._pagerank


    @property
    def connected_components(self):
        if self._connected_components is None:
            self._connected_components = self._calculate_connected_components()
        return self._connected_components


    @property
    def connected_components_counts(self):
        if self._connected_components_c is None:
            self._connected_components_c = self._calculate_connected_components_counts()
        return self._connected_components_c


    @property
    def nodes_summary(self):
        if self._nodes_summary is None:
            self._nodes_summary = self._calculate_nodes_summary()
        return self._nodes_summary


    @property
    def edges_summary(self):
        if self._edges_summary is None:
            self._edges_summary = self._calculate_edges_summary()
        return self._edges_summary


    @property
    def number_of_nodes(self):
        return self._number_of_nodes


    @property
    def number_of_edges(self):
        return self._number_of_edges


    def nodes_as_pandas(self):
        # TODO: This
        pass


    def edges_as_pandas(self):
        # TODO: This
        pass


    def nodes_as_dataframe(self):
        # TODO: This
        pass


    def edges_as_dataframe(self):
        # TODO: This
        pass


    def as_binary(self):
       # TODO: This
        pass


    def _from_pandas(self, edges, nodes, keys):
        # TODO: This
        pass


    def _from_dataframe(self, edges, nodes, keys):
        # TODO: This
        pass


    def _from_networkx(self, graph):
        # TODO: This
        pass


    def _from_graphframes(self, graph):
        # TODO: This
        pass


    def _from_binary(self, data):
        # TODO: This
        pass


    def _to_networkx(self):
        # TODO: This
        pass


    def _to_graphframe(self):
        # TODO: This
        pass


    def _to_dgl(self):
        # TODO: This
        pass


    def _calculate_degree(self):
        # TODO: This
        pass


    def _calculate_closeness_centrality(self):
        # TODO: This
        pass


    def _calculate_betweenness_centrality(self):
        # TODO: This
        pass


    def _calculate_pagerank(self):
        # TODO: This
        pass


    def _calculate_connected_components(self):
        # TODO: This
        pass


    def _calculate_connected_components_counts(self):
        # TODO: This
        pass


    def _calculate_nodes_summary(self):
        # TODO: This
        pass


    def _calculate_edges_summary(self):
        # TODO: This
        pass
