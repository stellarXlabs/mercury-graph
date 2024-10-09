import pytest

import pandas as pd
import networkx as nx

from mercury.graph.core.graph import Graph
from mercury.graph.core.spark_interface import SparkInterface, pyspark_installed, graphframes_installed


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


def test_graph():

	with pytest.raises(ValueError):
		graph = Graph()

	edges_df, keys, nodes_df = toy_datasets()

	graph = Graph(edges_df, keys, nodes_df)

	assert graph is not None

