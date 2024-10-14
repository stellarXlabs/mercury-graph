import pandas as pd

from mercury.graph.core import Graph
from mercury.graph.core.spark_interface import SparkInterface
from mercury.graph.graphml import SparkSpreadingActivation

# Create graph for testing
df_edges = pd.DataFrame(
    [
        ("A", "B", 5),
        ("A", "C", 2),
        ("B", "E", 8),
        ("C", "A", 1),
        ("D", "A", 8),
        ("D", "C", 2),
        ("E", "F", 3),
    ],
    columns=["src", "dst", "weight"],
)

df_nodes = pd.DataFrame(
    [
        ("A", "ALICE", 39),
        ("B", "BILL", 28),
        ("C", "CLAIR", 59),
        ("D", "DANIEL", 45),
        ("E", "ERIC", 90),
        ("F", "FANNY", 40),
    ],
    columns=["node_id", "name", "age"],
)

G = Graph(
    data=df_edges,
    nodes=df_nodes,
    keys={"src": "src", "dst": "dst", "weight": "weight", "id": "node_id"},
)


class TestSparkSpreadingActivation(object):

    def test_fit(self):

        spread_act = SparkSpreadingActivation(
            attribute="influence",
            spreading_factor=0.2,
            steps=2,
        )

        spread_act.fit(G, seed_nodes=["C", "D"])

        df_result = spread_act.updated_graph_.nodes_as_pandas()

        # Sum of total influence equals number of initial seed nodes
        assert round(df_result["influence"].sum(), 3) == 2.0
        # Influence doesn't reach E and F in 2 steps
        assert set(df_result[df_result["influence"] == 0]["id"].values) == {
            "E",
            "F",
        }
        assert set(df_result[df_result["influence"] > 0]["id"].values) == {
            "B",
            "D",
            "C",
            "A",
        }

    def test_fit_seed_nodes_as_df(self):
        """Test specifying the seed nodes as pyspark dataframe"""

        df_seed_nodes = pd.DataFrame([("C"), ("D"), ("E")], columns=["id"])
        df_seed_nodes = SparkInterface().spark.createDataFrame(df_seed_nodes)

        spread_act = SparkSpreadingActivation(
            attribute="influence",
            spreading_factor=0.2,
            steps=2,
        )

        g_spread = spread_act.fit(G, seed_nodes=df_seed_nodes)

        df_result = g_spread.updated_graph_.nodes_as_pandas()

        # Sum of total influence equals number of initial seed nodes
        assert round(df_result["influence"].sum(), 3) == 3.0
        # All nodes get influence in this case
        assert len(df_result[df_result["influence"] > 0]) == len(df_result)

    def test_fit_unweighted(self):

        spread_act = SparkSpreadingActivation(
            attribute="influence",
            spreading_factor=0.2,
            transfer_function="unweighted",
            steps=1,
        )

        g_spread = spread_act.fit(G, seed_nodes=["D"])

        df_result = g_spread.updated_graph_.nodes_as_pandas()

        assert round(df_result[df_result["id"] == "D"]["influence"].values[0], 3) == 0.8
        assert round(df_result[df_result["id"] == "C"]["influence"].values[0], 3) == 0.1
        assert round(df_result[df_result["id"] == "A"]["influence"].values[0], 3) == 0.1

    def test_fit_weighted(self):

        spread_act = SparkSpreadingActivation(
            attribute="influence",
            spreading_factor=0.2,
            transfer_function="weighted",
            steps=1,
        )

        g_spread = spread_act.fit(G, seed_nodes=["D"])

        df_result = g_spread.updated_graph_.nodes_as_pandas()

        assert round(df_result[df_result["id"] == "D"]["influence"].values[0], 3) == 0.8
        assert (
            round(df_result[df_result["id"] == "C"]["influence"].values[0], 3) == 0.04
        )
        assert (
            round(df_result[df_result["id"] == "A"]["influence"].values[0], 3) == 0.16
        )

    def test_fit_spreading_factor(self):

        spread_act = SparkSpreadingActivation(
            attribute="influence",
            spreading_factor=0.6,
            transfer_function="weighted",
            steps=1,
        )

        g_spread = spread_act.fit(
            G,
            seed_nodes=["D"],
        )

        df_result = g_spread.updated_graph_.nodes_as_pandas()

        assert round(df_result[df_result["id"] == "D"]["influence"].values[0], 3) == 0.4

    def test_fit_influenced_by(self):

        spread_act = SparkSpreadingActivation(
            attribute="influence",
            spreading_factor=0.2,
            steps=2,
            influenced_by=True,
        )

        g_spread = spread_act.fit(G, seed_nodes=["A", "D"])

        df_result = g_spread.updated_graph_.nodes_as_pandas()

        assert set(df_result[df_result["id"] == "A"]["influenced_by"].values[0]) == {
            "A",
            "D",
        }
        assert set(df_result[df_result["id"] == "B"]["influenced_by"].values[0]) == {
            "A",
            "D",
        }
        assert set(df_result[df_result["id"] == "C"]["influenced_by"].values[0]) == {
            "A",
            "D",
        }
        assert set(df_result[df_result["id"] == "D"]["influenced_by"].values[0]) == {
            "D"
        }
        assert set(df_result[df_result["id"] == "E"]["influenced_by"].values[0]) == {
            "A"
        }
        assert len(df_result[df_result["id"] == "F"]["influenced_by"].values[0]) == 0
