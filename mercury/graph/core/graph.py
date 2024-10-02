import inspect

import pandas as pd
import networkx as nx

from mercury.graph.core.spark_interface import SparkInterface, pyspark_installed, graphframes_installed, dgl_installed


class NodeIterator:
    """
    Iterator class for iterating over the nodes in a `mercury.graph.core.Graph`.

    This is returned by the `nodes` property of the `mercury.graph.core.Graph` class.

    Args:
        graph: The `mercury.graph.core.Graph` object to iterate over.

    Usage:

    ```python
    g = mg.Graph(data)
    for node in g.nodes:
        print(node)
    ```
    """

    def __init__(self, graph):
        self.graph = graph
        self.ix = -1
        if graph._as_networkx is not None:
            self.key_list = list(graph._as_networkx.nodes.keys())
        else:
            self.key_list = [row['id'] for row in graph.graphframe.vertices.select('id').collect()]


    def __iter__(self):
        return self


    def __next__(self):
        self.ix += 1

        if self.ix >= len(self.key_list):
            raise StopIteration

        ky = self.key_list[self.ix]

        if self.graph._as_networkx is not None:
            d = self.graph._as_networkx.nodes[ky]
            d['id'] = ky

            return d

        g = self.graph.graphframe
        return g.vertices.filter(g.vertices.id == ky).first().asDict()


class EdgeIterator:
    """
    Iterator class for iterating over the edges in a `mercury.graph.core.Graph`.

    This is returned by the `edges` property of the `mercury.graph.core.Graph` class.

    Args:
        graph: The `mercury.graph.core.Graph` object to iterate over.

    Usage:

    ```python
    g = mg.Graph(data)
    for edge in g.edges:
        print(edge)
    ```
    """

    def __init__(self, graph):
        self.graph = graph
        self.ix = -1
        if graph._as_networkx is not None:
            self.key_list = list(graph._as_networkx.edges.keys())
        else:
            self.key_list = [(row['src'], row['dst']) for row in graph.graphframe.edges.collect()]


    def __iter__(self):
        return self


    def __next__(self):
        self.ix += 1

        if self.ix >= len(self.key_list):
            raise StopIteration

        ky = self.key_list[self.ix]

        if self.graph._as_networkx is not None:
            d = self.graph._as_networkx.edges[ky]
            d['src'] = ky[0]
            d['dst'] = ky[1]

            return d

        g = self.graph.graphframe
        return g.edges.filter((g.edges.src == ky[0]) & (g.edges.dst == ky[1])).first().asDict()


class Graph:
    """ This is the main class in mercury.graph.

    This class seamlessly abstracts the underlying technology used to represent the graph. You can create a graph passing the following
    objects to the constructor:

    - A pandas DataFrame containing edges (with a keys dictionary to specify the columns and possibly a nodes DataFrame)
    - A pyspark DataFrame containing edges (with a keys dictionary to specify the columns and possibly a nodes DataFrame)
    - A networkx graph
    - A graphframes graph

    Bear in mind that the graph object is immutable. This means that you can't modify the graph object once it has been created. If you
    want to modify it, you have to create a new graph object.

    The graph object provides:

    - Properties to access the graph in different formats (networkx, graphframes, dgl)
    - Properties with metrics and summary information that are calculated on demand and technology independent.
    - It is inherited by other graph classes in mercury-graph providing ML algorithms such as graph embedding, visualization, etc.

    Args:
		data: The data to create the graph from. It can be a pandas DataFrame, a networkx Graph, a pyspark DataFrame, or a Graphframe. In
        	case it already contains a graph (networkx or graphframes), the keys and nodes arguments are ignored.
		keys: A dictionary with keys to specify the columns in the data DataFrame. The keys are:
			- 'src': The name of the column with the source node.
			- 'dst': The name of the column with the destination node.
			- 'id': The name of the column with the node id.
			- 'weight': The name of the column with the edge weight.
			- 'directed': A boolean to specify if the graph is directed. (Only for pyspark DataFrames)
            When the keys argument is not provided or the key is missing, the default values are:
            - 'src': 'src'
            - 'dst': 'dst'
            - 'id': 'id'
            - 'weight': 'weight'
            - 'directed': True
		nodes: A pandas DataFrame or a pyspark DataFrame with the nodes data. (Only when `data` is pandas or pyspark DataFrame and with the
        	same type as `data`) If not given, the nodes are inferred from the edges DataFrame.
    """
    def __init__(self, data = None, keys = None, nodes = None):
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
        self._is_directed = False
        self._is_weighted = False

        self._init_values = {k: v for k, v in locals().items() if k in inspect.signature(self.__init__).parameters}

        if type(data) == pd.core.frame.DataFrame:
            self._from_pandas(data, nodes, keys)
            return

        if isinstance(data, nx.Graph):      # This is the most general case, including: ...Graph, ...DiGraph and ...MultiGraph
            self._from_networkx(data)
            return

        spark_int = SparkInterface()

        if pyspark_installed and type(data) == spark_int.type_spark_dataframe:
            self._from_dataframe(data, nodes, keys)
            return

        if graphframes_installed and type(data) == spark_int.type_graphframe:
            self._from_graphframes(data)
            return

        raise ValueError('Invalid input data. (Expected: pandas DataFrame, a networkx Graph, a pyspark DataFrame, a graphframes Graph.)')


    def __str__(self):
        txt = []
        txt.append('mercury.graph.core.Graph with %d nodes and %d edges.' % (self._number_of_nodes, self._number_of_edges))
        txt.append('  is_directed:    %s' % self._is_directed)
        txt.append('  is_weighted:    %s' % self._is_weighted)
        txt.append('  has_networkx:   %s' % (self._as_networkx is not None))
        txt.append('  has_graphframe: %s' % (self._as_graphframe is not None))
        txt.append('  has_dgl:        %s' % (self._as_dgl is not None))

        return '\n'.join(txt)


    def __repr__(self):
        return 'Graph(%s)' % ', '.join('%s = %s' % (k, v) for k, v in self._init_values.items())


    @property
    def nodes(self):
        """
        Returns an iterator over all the nodes in the graph.

        Returns:
            NodeIterator: An iterator that yields each node in the graph.
        """
        return NodeIterator(self)


    @property
    def edges(self):
        """
        Returns an iterator over the edges in the graph.

        Returns:
            EdgeIterator: An iterator object that allows iterating over the edges in the graph.
        """
        return EdgeIterator(self)


    @property
    def networkx(self):
        """
        Returns the graph representation as a NetworkX graph.

        If the graph has not been converted to NetworkX format yet, it will be converted and cached for future use.

        Returns:
            networkx.Graph: The graph representation as a NetworkX graph.
        """
        if self._as_networkx is None:
            self._as_networkx = self._to_networkx()

        return self._as_networkx


    @property
    def graphframe(self):
        """
        Returns the graph as a GraphFrame.

        If the graph has not been converted to a GraphFrame yet, it will be converted and cached for future use.

        Returns:
            GraphFrame: The graph represented as a GraphFrame.
        """
        if self._as_graphframe is None:
            self._as_graphframe = self._to_graphframe()

        return self._as_graphframe


    @property
    def dgl(self):
        """
        Returns the graph as a DGL graph.

        If the graph has not been converted to a DGL graph yet, it will be converted and cached for future use.

        Returns:
            dgl.DGLGraph: The graph represented as a DGL graph.
        """
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
        """
        Returns the number of nodes in the graph.

        Returns:
            int: The number of nodes in the graph.
        """
        return self._number_of_nodes


    @property
    def number_of_edges(self):
        """
        Returns the number of edges in the graph.

        Returns:
            int: The number of edges in the graph.
        """
        return self._number_of_edges


    @property
    def is_directed(self):
        """
        Returns True if the graph is directed, False otherwise.

        Note: Graphs created using graphframes are always directed. The way around it is to add the reverse edges to the graph.
        This can be done by creating the Graph with pyspark DataFrame() and defining a key 'directed' set as False in the `dict`
        argument. Otherwise, the graph will be considered directed even if these reversed edges have been created by other means
        this class cannot be aware of.
        """
        return self._is_directed


    @property
    def is_weighted(self):
        """
        Returns True if the graph is weighted, False otherwise.

        A graph is considered weight if it has a column named 'weight' in the edges DataFrame or the column has a different name and that
        name is passed in the `dict` argument as the 'weight' key.
        """
        return self._is_weighted


    def nodes_as_pandas(self):
        """
        Returns the nodes as a pandas DataFrame.

        If the graph is represented as a networkx graph, the nodes are extracted from it. Otherwise, the graphframes graph will be used.
        This dataset may differ from possible pandas DataFrame passed to the constructor in the column names and order. The column used
        as the node id is always named 'id'.
        """
        if self._as_networkx is not None:
            nodes_data = self._as_networkx.nodes(data = True)
            nodes_df   = pd.DataFrame([(node, attr) for node, attr in nodes_data], columns = ['id', 'attributes'])
            attrs_df   = pd.json_normalize(nodes_df['attributes'])

            return pd.concat([nodes_df.drop('attributes', axis = 1), attrs_df], axis = 1)

        return self.graphframe.vertices.toPandas()


    def edges_as_pandas(self):
        """
        Returns the edges as a pandas DataFrame.

        If the graph is represented as a networkx graph, the edges are extracted from it. Otherwise, the graphframes graph will be used.
        This dataset may differ from possible pandas DataFrame passed to the constructor in the column names and order. The columns used
        as the source and destination nodes are always named 'src' and 'dst', respectively.
        """
        if self._as_networkx is not None:
            edges_data = self._as_networkx.edges(data = True)
            edges_df   = pd.DataFrame([(src, dst, attr) for src, dst, attr in edges_data], columns = ['src', 'dst', 'attributes'])
            attrs_df   = pd.json_normalize(edges_df['attributes'])

            return pd.concat([edges_df.drop('attributes', axis = 1), attrs_df], axis = 1)

        return self.graphframe.edges.toPandas()


    def nodes_as_dataframe(self):
        """
        Returns the nodes as a pyspark DataFrame.

        If the graph is represented as a graphframes graph, the nodes are extracted from it. Otherwise, the nodes are converted from the
        pandas DataFrame representation. The column used as the node id is always named 'id', regardless of the original column name passed
        to the constructor.
        """
        if self._as_graphframe is not None:
            return self._as_graphframe.vertices

        return SparkInterface().spark.createDataFrame(self.nodes_as_pandas())


    def edges_as_dataframe(self):
        """
        Returns the edges as a pyspark DataFrame.

        If the graph is represented as a graphframes graph, the edges are extracted from it. Otherwise, the edges are converted from the
        pandas DataFrame representation. The columns used as the source and destination nodes are always named 'src' and 'dst',
        respectively, regardless of the original column names passed to the constructor.
        """
        if self._as_graphframe is not None:
            return self._as_graphframe.edges

        return SparkInterface().spark.createDataFrame(self.edges_as_pandas())


    def _from_pandas(self, edges, nodes, keys):
        """ This internal method extends the constructor to accept a pandas DataFrame as input.

        It takes the constructor arguments and does not return anything. It sets the internal state of the object.
        """
        if keys is None:
            src = 'src'
            dst = 'dst'
            id  = 'id'
            weight = 'weight'
            directed = True
        else:
            src = keys.get('src', 'src')
            dst = keys.get('dst', 'dst')
            id  = keys.get('id', 'id')
            weight = keys.get('weight', 'weight')
            directed = keys.get('directed', True)

        if directed:
            g = nx.DiGraph()
        else:
            g = nx.Graph()

        self._is_weighted = weight in edges.columns

        for _, row in edges.iterrows():
            attr = row.drop([src, dst]).to_dict()
            g.add_edge(row[src], row[dst], **attr)

        if nodes is not None:
            for _, row in nodes.iterrows():
                attr = row.drop([id]).to_dict()
                g.add_node(row[id], **attr)

        self._from_networkx(g)


    def _from_dataframe(self, edges, nodes, keys):
        """ This internal method extends the constructor to accept a pyspark DataFrame as input.

        It takes the constructor arguments and does not return anything. It sets the internal state of the object.
        """
        if not graphframes_installed:
            raise ImportError('graphframes is not installed')

        if keys is None:
            directed = True
        else:
            src = keys.get('src', 'src')
            dst = keys.get('dst', 'dst')
            id  = keys.get('id', 'id')
            weight = keys.get('weight', 'weight')
            directed = keys.get('directed', True)

            edges = edges.withColumnRenamed(src, 'src').withColumnRenamed(dst, 'dst')

            if weight in edges.columns:
                self._is_weighted = True
                edges = edges.withColumnRenamed(weight, 'weight')

            if nodes is not None:
                nodes = nodes.withColumnRenamed(id, 'id')
            else:
                src_nodes = edges.select(src).distinct().withColumnRenamed(src, id)
                dst_nodes = edges.select(dst).distinct().withColumnRenamed(dst, id)
                nodes = src_nodes.union(dst_nodes).distinct()

        g = SparkInterface().graphframes.GraphFrame(nodes, edges)

        if not directed:
            edges = g.edges

            other_columns = [col for col in edges.columns if col not in ('src', 'dst')]
            reverse_edges = edges.select(edges['dst'].alias('src'), edges['src'].alias('dst'), *other_columns)
            all_edges     = edges.union(reverse_edges).distinct()

            g = SparkInterface().graphframes.GraphFrame(nodes, all_edges)

        self._from_graphframes(g, directed)


    def _from_networkx(self, graph):
        """ This internal method extends the constructor to accept a networkx graph as input.

        It takes the constructor arguments and does not return anything. It sets the internal state of the object.
        """
        self._as_networkx = graph
        self._number_of_nodes = len(graph.nodes)
        self._number_of_edges = len(graph.edges)
        self._is_directed = nx.is_directed(graph)


    def _from_graphframes(self, graph, directed = True):
        """ This internal method extends the constructor to accept a graphframes graph as input.

        It takes the constructor arguments and does not return anything. It sets the internal state of the object.
        """
        self._as_graphframe = graph
        self._number_of_nodes = graph.vertices.count()
        self._number_of_edges = graph.edges.count()
        self._is_directed = directed


    def _to_networkx(self):
        """ This internal method handles the logic of a property. It returns the networkx graph that already exists
        or converts it from the graphframes graph if not."""
        if self._as_networkx is None:
            self._as_networkx = nx.DiGraph()

            for _, row in self.nodes_as_pandas().iterrows():
                attr = row.drop(['src', 'dst']).to_dict()
                self._as_networkx.add_edge(row['src'], row['dst'], **attr)

            for _, row in self.edges_as_pandas().iterrows():
                attr = row.drop(['id']).to_dict()
                self._as_networkx.add_node(row['id'], **attr)

        return self._as_networkx


    def _to_graphframe(self):
        """ This internal method handles the logic of a property. It returns the graphframes graph that already exists
        or converts it from the networkx graph if not."""
        if self._as_graphframe is None:
            nodes = self.nodes_as_dataframe()
            edges = self.edges_as_dataframe()

            self._as_graphframe = SparkInterface().graphframes.GraphFrame(nodes, edges)

        return self._as_graphframe


    def _to_dgl(self):
        """ This internal method handles the logic of a property. It returns the dgl graph that already exists
        or converts it from the networkx graph if not."""
        if self._as_dgl is None and dgl_installed:
            dgl = SparkInterface().dgl

            edge_attrs = [c for c in self.networkx.columns if c not in ['src', 'dst']]
            if len(edge_attrs) > 0:
                edge_attrs = None

            node_attrs = [c for c in self.networkx.columns if c not in ['id']]
            if len(node_attrs) > 0:
                node_attrs = None

            self._as_dgl = dgl.from_networkx(self.networkx, edge_attrs = edge_attrs, node_attrs = node_attrs)

        return self._as_dgl


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
