import pytest
import pandas as pd

from mercury.graph.core import Graph
from mercury.graph.ml import SpectralClustering


class TestSpectral(object):
    def test_instancing(object):
        """
        Tests instancing of the class SpectralClustering
        """

        spectral_clustering = SpectralClustering()
        assert isinstance(spectral_clustering, SpectralClustering)

        assert (
            type(str(spectral_clustering)) is str and len(str(spectral_clustering)) > 0
        )
        assert (
            type(repr(spectral_clustering)) is str
            and len(repr(spectral_clustering)) > 0
        )

    def test___init__(self):
        """
        Tests method SpectralClustering.__init__
        """
        spectral_clustering = SpectralClustering(4)
        assert spectral_clustering.n_clusters == 4

        expected_msg = "Error: Mode must be either 'networkx' or 'spark'"
        with pytest.raises(ValueError, match=expected_msg):
            spectral_clustering = SpectralClustering(4, mode="wrong")

    def test_fit(self):
        """
        Tests method SpectralClustering.fit
        """

        df_edges = pd.DataFrame(
            {
                "src": ["a", "a", "a", "a", "b", "c", "e", "d", "d", "d", "g", "h", "f", "j", "j", "i"],
                "dst": ["b", "c", "e", "z", "c", "e", "d", "g", "f", "h", "f", "f", "j", "i", "l", "l"],
                "value": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            }
        )

        assert df_edges.shape[0] == 16

        df_nodes = pd.DataFrame(
            {
                "node_id": ["a", "b", "c", "e", "d", "f", "g", "h", "j", "i", "z", "l"],
            }
        )

        assert df_nodes.shape[0] == 12

        g = Graph(
            data=df_edges,
            nodes=df_nodes,
            keys={"src": "src", "dst": "dst", "weight": "value", "id": "node_id"},
        )

        assert isinstance(g, Graph)

        spectral_clustering = SpectralClustering(3)

        len_str = len(str(spectral_clustering))
        spectral_clustering.fit(g)
        len_str_fit = len(str(spectral_clustering))
        assert len_str_fit > len_str

        assert spectral_clustering.labels_.shape[0] == 12
        assert (
            spectral_clustering.labels_.cluster[
                spectral_clustering.labels_.node_id == "a"
            ].values[0]
            == spectral_clustering.labels_.cluster[
                spectral_clustering.labels_.node_id == "b"
            ].values[0]
        )
        assert (
            spectral_clustering.labels_.cluster[
                spectral_clustering.labels_.node_id == "a"
            ].values[0]
            == spectral_clustering.labels_.cluster[
                spectral_clustering.labels_.node_id == "c"
            ].values[0]
        )
        assert (
            spectral_clustering.labels_.cluster[
                spectral_clustering.labels_.node_id == "a"
            ].values[0]
            == spectral_clustering.labels_.cluster[
                spectral_clustering.labels_.node_id == "e"
            ].values[0]
        )

    def test_fit_spark(self):
        """
        Tests method SpectralClustering.fit
        """

        df_edges = pd.DataFrame(
            {
                "src": ["a", "a", "a", "a", "b", "c", "e", "d", "d", "d", "g", "h", "f", "j", "j", "i"],
                "dst": ["b", "c", "e", "z", "c", "e", "d", "g", "f", "h", "f", "f", "j", "i", "l", "l"],
                "value": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            }
        )

        assert df_edges.shape[0] == 16

        df_nodes = pd.DataFrame(
            {
                "node_id": ["a", "b", "c", "e", "d", "f", "g", "h", "j", "i", "z", "l"],
            }
        )

        assert df_nodes.shape[0] == 12

        g = Graph(
            data=df_edges,
            nodes=df_nodes,
            keys={"src": "src", "dst": "dst", "weight": "value", "id": "node_id"},
        )

        assert isinstance(g, Graph)

        spectral_clustering = SpectralClustering(3, mode="spark", max_iterations=8)

        spectral_clustering.fit(g)

        labels_ = spectral_clustering.labels_.toPandas()

        assert labels_.shape == (12, 2)
        assert (
            labels_["node_id"].sort_values().reset_index(drop=True)
            .equals(df_nodes["node_id"].sort_values().reset_index(drop=True))
        )
        assert sorted(labels_["cluster"].unique()) == [0, 1, 2]

        assert -0.5 <= spectral_clustering.modularity_ <= 1
