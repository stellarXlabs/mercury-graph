import pytest

from unittest.mock import patch, MagicMock

import pandas as pd
import networkx as nx

from mercury.graph.core.graph import Graph, NodeIterator, EdgeIterator
from mercury.graph.core.spark_interface import SparkInterface, pyspark_installed, graphframes_installed, dgl_installed


def toy_datasets():
    data = {
        'Person_A': ['Alice', 'Bob', 'Alice', 'Eve', 'Diana', 'Charlie', 'Frank', 'Bob', 'Grace', 'Alice'],
        'Person_B': ['Bob', 'Charlie', 'Diana', 'Frank', 'Eve', 'Grace', 'Grace', 'Eve', 'Diana', 'Frank'],
        'Duration': [10, 15, 5, 20, 12, 8, 7, 14, 10, 25]
    }

    node_data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace'],
        'Role': ['Manager', 'Engineer', 'Engineer', 'HR', 'Manager', 'Technician', 'Technician'],
        'Age':  [34, 28, 30, 25, 40, 35, 29]
    }

    keys = {'id': 'Name', 'src': 'Person_A', 'dst': 'Person_B', 'weight': 'Duration'}

    return pd.DataFrame(data), keys, pd.DataFrame(node_data)


def check_graph_content(g, short = False, directed = True, weighted = True, duplicate = 1):
    assert g.number_of_nodes == 7
    assert g.number_of_edges == 10*duplicate
    assert g.is_directed == directed
    assert g.is_weighted == weighted

    is_nx = g._as_networkx is not None
    is_gf = g._as_graphframe is not None

    if not short and is_gf and not is_nx:
        nn = g.nodes_colnames
        assert type(nn) == list
        assert len(nn) == 3
        assert 'id' in nn

        nn = g.edges_colnames
        assert type(nn) == list
        assert len(nn) == 3
        assert 'src' in nn
        assert 'dst' in nn

        cc = g.closeness_centrality
        assert type(cc) == dict
        assert len(cc) == 7
        assert cc['Alice'] >= 0

        pr = g.pagerank
        assert type(pr) == dict
        assert len(pr) == 7
        assert pr['Alice'] >= 0

        cc = g.connected_components
        assert type(cc) == dict
        assert len(cc) == 7
        assert type(cc['Alice']) == dict
        assert len(cc['Alice']) == 2
        assert cc['Alice']['cc_size'] == 7
        assert cc['Alice']['cc_id'] == cc['Diana']['cc_id']

    ni = g.nodes

    assert type(ni) == NodeIterator
    n = 0
    for node in ni:
        n += 1
        assert type(node) == dict
        assert 'id' in node
        if not short:
            assert len(node) == 3
            assert 'Role' in node
            assert 'Age' in node

    assert n == 7

    ei = g.edges

    assert type(ei) == EdgeIterator
    n = 0
    for edge in ei:
        n += 1
        assert type(edge) == dict
        assert 'src' in edge
        assert 'dst' in edge
        if not short:
            assert len(edge) == 3
            assert 'weight' in edge

    assert n == 10*duplicate

    assert is_nx == (g._as_networkx is not None)
    assert is_gf == (g._as_graphframe is not None)

    df = g.nodes_as_pandas()
    assert df.shape[0] == 7
    if not short:
        assert df.shape[1] == 3

    df = g.edges_as_pandas()
    assert df.shape[0] == 10*duplicate
    if not short:
        assert df.shape[1] == 3

    assert is_nx == (g._as_networkx is not None)
    assert is_gf == (g._as_graphframe is not None)

    deg = g.degree
    assert type(deg) == dict
    assert len(deg) == 7
    assert deg['Alice'] == 3*duplicate

    if directed:
        deg = g.in_degree
        assert type(deg) == dict
        assert len(deg) == 7
        assert deg['Alice'] == 0

        deg = g.out_degree
        assert type(deg) == dict
        assert len(deg) == 7
        assert deg['Alice'] == 3

    assert is_nx == (g._as_networkx is not None)
    assert is_gf == (g._as_graphframe is not None)

    bc = g.betweenness_centrality
    assert type(bc) == dict
    assert len(bc) == 7
    assert bc['Alice'] >= 0

    assert g._as_networkx is not None
    assert is_gf == (g._as_graphframe is not None)

    cc = g.closeness_centrality
    assert type(cc) == dict
    assert len(cc) == 7
    assert cc['Alice'] >= 0

    pr = g.pagerank
    assert type(pr) == dict
    assert len(pr) == 7
    assert pr['Alice'] >= 0

    cc = g.connected_components
    assert type(cc) == dict
    assert len(cc) == 7
    assert type(cc['Alice']) == dict
    assert len(cc['Alice']) == 2
    assert cc['Alice']['cc_size'] == 7
    assert cc['Alice']['cc_id'] == cc['Diana']['cc_id']

    if short:
        return

    nn = g.nodes_colnames
    assert type(nn) == list
    assert len(nn) == 3
    assert 'id' in nn

    nn = g.edges_colnames
    assert type(nn) == list
    assert len(nn) == 3
    assert 'src' in nn
    assert 'dst' in nn

    df = g.nodes_as_pandas()
    assert df.shape == (7, 3)

    df = g.edges_as_pandas()
    assert df.shape == (10*duplicate, 3)


def do_spark_parts(g, nodes_df, edges_df, keys):
    # Test from a graphframe with everything

    g = Graph(g.graphframe)

    assert g is not None
    assert g._as_graphframe is not None and g._as_networkx is None

    df = g.nodes_as_pandas()
    assert df.shape == (7, 3)
    df = g.edges_as_pandas()
    assert df.shape == (10, 3)

    s = str(g).replace(' ', '').replace('\n', '')
    assert '7nodesand10edges' in s
    assert 's_directed:Trueis_weighted:Truehas_networkx:Falsehas_graphframe:Truehas_dgl:False' in s

    s = repr(g).replace(' ', '').replace('\n', '')
    assert 'data=GraphFrame(' in s
    assert 'keys=None' in s
    assert 'nodes=None' in s

    check_graph_content(g)

    df = g.nodes_as_dataframe()
    assert len(df.columns) == 3
    assert df.count() == 7

    df = g.edges_as_dataframe()
    assert len(df.columns) == 3
    assert df.count() == 10

    # Test from a pyspark dataframe with everything

    spark = SparkInterface().spark

    nodes_spark = spark.createDataFrame(nodes_df)
    edges_spark = spark.createDataFrame(edges_df)

    g = Graph(edges_spark, keys, nodes_spark)

    assert g is not None
    assert g._as_graphframe is not None and g._as_networkx is None

    df = g.nodes_as_pandas()
    assert df.shape == (7, 3)
    df = g.edges_as_pandas()
    assert df.shape == (10, 3)

    s = str(g).replace(' ', '').replace('\n', '')
    assert '7nodesand10edges' in s
    assert 's_directed:Trueis_weighted:Truehas_networkx:Falsehas_graphframe:Truehas_dgl:False' in s

    s = repr(g).replace(' ', '').replace('\n', '')
    assert 'data=DataFrame[Person_A' in s
    assert 'keys={\'id' in s
    assert 'nodes=DataFrame[Name' in s

    check_graph_content(g)

    # Test from a pyspark dataframe without node attributes nor keys

    df_spark = edges_spark.select('Person_A', 'Person_B').withColumnRenamed('Person_A', 'src').withColumnRenamed('Person_B', 'dst')
    g = Graph(df_spark)

    assert g is not None
    assert g._as_graphframe is not None and g._as_networkx is None

    df = g.nodes_as_pandas()
    assert df.shape == (7, 1)
    df = g.edges_as_pandas()
    assert df.shape == (10, 2)

    s = str(g).replace(' ', '').replace('\n', '')
    assert '7nodesand10edges' in s
    assert 's_directed:Trueis_weighted:Falsehas_networkx:Falsehas_graphframe:Truehas_dgl:False' in s

    s = repr(g).replace(' ', '').replace('\n', '')
    assert 'data=DataFrame[src' in s
    assert 'keys=None' in s
    assert 'nodes=None' in s

    check_graph_content(g, short = True, weighted = False)

    # Test from a pyspark dataframe but not directed

    g = Graph(df_spark, keys = {'directed': False})

    assert g is not None
    assert g._as_graphframe is not None and g._as_networkx is None

    df = g.nodes_as_pandas()
    assert df.shape == (7, 1)
    df = g.edges_as_pandas()
    assert df.shape == (20, 2)

    s = str(g).replace(' ', '').replace('\n', '')
    assert '7nodesand20edges' in s
    assert 's_directed:Falseis_weighted:Falsehas_networkx:Falsehas_graphframe:Truehas_dgl:False' in s

    s = repr(g).replace(' ', '').replace('\n', '')
    assert 'data=DataFrame[src' in s
    assert 'keys={\'directed' in s
    assert 'nodes=None' in s

    assert g.is_directed == False

    check_graph_content(g, short = True, directed = False, weighted = False, duplicate = 2)

    gnx = g.networkx

    assert gnx.is_directed() == False

    g = Graph(gnx)  # Conversion to NetworkX made the edges unique, therefore not duplicated anymore.

    check_graph_content(g, short = True, directed = False, weighted = False)


def test_graph():

    with pytest.raises(ValueError):
        graph = Graph()

    edges_df, keys, nodes_df = toy_datasets()

    # Test with pandas dataframes all attributes and keys

    g = Graph(edges_df, keys, nodes_df)

    assert g is not None
    assert g._as_graphframe is None and g._as_networkx is not None

    df = g.nodes_as_pandas()
    assert df.shape == (7, 3)
    df = g.edges_as_pandas()
    assert df.shape == (10, 3)

    s = str(g).replace(' ', '').replace('\n', '')
    assert '7nodesand10edges' in s
    assert 's_directed:Trueis_weighted:Truehas_networkx:Truehas_graphframe:Falsehas_dgl:False' in s

    s = repr(g).replace(' ', '').replace('\n', '')
    assert 'data=Person_A' in s
    assert 'keys={\'id\':' in s
    assert 'nodes=NameRol' in s

    check_graph_content(g)

    # Test from networkx graph with everything

    g = Graph(g.networkx)

    assert g is not None
    assert g._as_graphframe is None and g._as_networkx is not None

    df = g.nodes_as_pandas()
    assert df.shape == (7, 3)
    df = g.edges_as_pandas()
    assert df.shape == (10, 3)

    s = str(g).replace(' ', '').replace('\n', '')
    assert '7nodesand10edges' in s
    assert 's_directed:Trueis_weighted:Truehas_networkx:Truehas_graphframe:Falsehas_dgl:False' in s

    s = repr(g).replace(' ', '').replace('\n', '')
    assert 'data=DiGraph' in s
    assert 'keys=None' in s
    assert 'nodes=None' in s

    check_graph_content(g)

    # Test from pandas without node attributes

    g2 = Graph(edges_df, keys)

    assert g2 is not None
    assert g2._as_graphframe is None and g2._as_networkx is not None

    df = g2.nodes_as_pandas()
    assert df.shape == (7, 1)
    df = g2.edges_as_pandas()
    assert df.shape == (10, 3)

    s = str(g2).replace(' ', '').replace('\n', '')
    assert '7nodesand10edges' in s
    assert 's_directed:Trueis_weighted:Truehas_networkx:Truehas_graphframe:Falsehas_dgl:False' in s

    s = repr(g2).replace(' ', '').replace('\n', '')
    assert 'data=Person_A' in s
    assert 'keys={\'id\':' in s
    assert 'nodes=None' in s

    check_graph_content(g2, short = True)

    # Test from pandas without node attributes and keys

    df = edges_df.copy()
    df.columns = ['src', 'dst', 'weight']
    df.drop('weight', axis = 1, inplace = True)
    g2 = Graph(df)

    assert g2 is not None
    assert g2._as_graphframe is None and g2._as_networkx is not None

    df = g2.nodes_as_pandas()
    assert df.shape == (7, 1)
    df = g2.edges_as_pandas()
    assert df.shape == (10, 2)

    s = str(g2).replace(' ', '').replace('\n', '')
    assert '7nodesand10edges' in s
    assert 's_directed:Trueis_weighted:Falsehas_networkx:Truehas_graphframe:Falsehas_dgl:False' in s

    s = repr(g2).replace(' ', '').replace('\n', '')
    assert 'data=srcdst' in s
    assert 'keys=None' in s
    assert 'nodes=None' in s

    check_graph_content(g2, short = True, weighted = False)

    # Test from networkx graph not directed

    nd_keys = keys.copy()
    nd_keys['directed'] = False
    ng = Graph(edges_df, nd_keys, nodes_df)

    assert ng is not None
    assert ng._as_graphframe is None and ng._as_networkx is not None

    df = ng.nodes_as_pandas()
    assert df.shape == (7, 3)
    df = ng.edges_as_pandas()
    assert df.shape == (10, 3)

    s = str(ng).replace(' ', '').replace('\n', '')
    assert '7nodesand10edges' in s
    assert 's_directed:Falseis_weighted:Truehas_networkx:Truehas_graphframe:Falsehas_dgl:False' in s

    s = repr(ng).replace(' ', '').replace('\n', '')
    assert 'data=Person_A' in s
    assert 'keys={\'id\':' in s
    assert 'nodes=NameRol' in s

    assert ng.is_directed == False

    check_graph_content(ng, short = True, directed = False)

    # Test pyspark and graphframes

    if pyspark_installed and graphframes_installed:
        do_spark_parts(g, nodes_df, edges_df, keys)


def mock_from_networkx(networkx, edge_attrs = None, node_attrs = None):
    ret = {'networkx': networkx, 'edge_attrs': edge_attrs, 'node_attrs': node_attrs}
    return ret


mock_dgl = MagicMock()
mock_dgl.from_networkx = mock_from_networkx


def test_graph_to_dgl():
    edges_df, keys, nodes_df = toy_datasets()
    g = Graph(edges_df, keys, nodes_df)

    if not dgl_installed:
        assert g.dgl is None

    with patch.dict('sys.modules', {'dgl': mock_dgl}):
        import mercury.graph.core.spark_interface as sp_int
        import mercury.graph.core.graph as core_graph

        bak_dgl_installed = sp_int.dgl_installed
        bak_sp_int_dgl    = SparkInterface._dgl

        sp_int.dgl_installed     = True
        core_graph.dgl_installed = True
        SparkInterface._dgl      = mock_dgl

        # Here starts the test with the mocked library.

        g = Graph(edges_df, keys, nodes_df)

        mock_ret = g.dgl

        assert type(mock_ret) == dict

        assert type(mock_ret['networkx']) == nx.DiGraph

        assert type(mock_ret['edge_attrs']) == list
        assert len(mock_ret['edge_attrs']) == 1
        assert mock_ret['edge_attrs'][0] == 'weight'

        assert type(mock_ret['node_attrs']) == list
        assert len(mock_ret['node_attrs']) == 2
        assert 'Role' in mock_ret['node_attrs']
        assert 'Age' in mock_ret['node_attrs']

        # Same without attributes

        df = edges_df.copy()
        df.columns = ['src', 'dst', 'weight']
        df.drop('weight', axis = 1, inplace = True)
        g = Graph(df)

        mock_ret = g.dgl

        assert type(mock_ret) == dict

        assert type(mock_ret['networkx']) == nx.DiGraph
        assert mock_ret['edge_attrs'] is None
        assert mock_ret['node_attrs'] is None

        # We restore the state of mercury.graph.core.spark_interface undoing the mocking

        sp_int.dgl_installed     = bak_dgl_installed
        core_graph.dgl_installed = bak_dgl_installed
        SparkInterface._dgl      = bak_sp_int_dgl


# test_graph()
# test_graph_to_dgl()
