import pandas as pd

from mercury.graph.core import Graph
from mercury.graph.graphml import SpectralClustering


class TestSpectral(object):
    def test_instancing(object):
        """
        Tests instancing of the class SpectralClustering
        """

        spectral_clustering = SpectralClustering()

        assert isinstance(spectral_clustering, SpectralClustering)


    def test___init__(self):
        """
        Tests method SpectralClustering.__init__
        """
        spectral_clustering = SpectralClustering(4)

        assert spectral_clustering.n_clusters == 4


    def test_fit(self):
        """
        Tests method SpectralClustering.fit
        """

        df_edges = pd.DataFrame({
            'src':['a', 'a', 'a', 'a', 'b', 'c', 'e', 'd', 'd', 'd', 'g', 'h', 'f', 'j', 'j', 'i'],
            'dst':['b', 'c', 'e', 'z', 'c', 'e', 'd', 'g', 'f', 'h', 'f', 'f', 'j', 'i', 'l', 'l' ],
            'value':[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        })

        assert df_edges.shape[0] == 16

        df_nodes = pd.DataFrame({
            'node_id':['a', 'b', 'c', 'e', 'd', 'f', 'g', 'h', 'j', 'i', 'z', 'l'],
        })

        assert df_nodes.shape[0] == 12

        g = Graph(data=df_edges,
                  nodes=df_nodes,
                  keys={"src": "src",
                        "dst": "dst",
                        "weight": "value",
                        "id": "node_id"})

        assert isinstance(g, Graph)

        spectral_clustering = SpectralClustering(3)

        spectral_clustering.fit(g)

        assert spectral_clustering.labels_.shape[0] == 12
        assert spectral_clustering.labels_.cluster[spectral_clustering.labels_.node_id == 'a'].values[0] == spectral_clustering.labels_.cluster[spectral_clustering.labels_.node_id == 'b'].values[0]
        assert spectral_clustering.labels_.cluster[spectral_clustering.labels_.node_id == 'a'].values[0] == spectral_clustering.labels_.cluster[spectral_clustering.labels_.node_id == 'c'].values[0]
        assert spectral_clustering.labels_.cluster[spectral_clustering.labels_.node_id == 'a'].values[0] == spectral_clustering.labels_.cluster[spectral_clustering.labels_.node_id == 'e'].values[0]


    def test_fit_spark(self):
        """
        Tests method SpectralClustering.fit
        """

        df_edges = pd.DataFrame({
            'src':['a', 'a', 'a', 'a', 'b', 'c', 'e', 'd', 'd', 'd', 'g', 'h', 'f', 'j', 'j', 'i'],
            'dst':['b', 'c', 'e', 'z', 'c', 'e', 'd', 'g', 'f', 'h', 'f', 'f', 'j', 'i', 'l', 'l' ],
            'value':[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
            })

        assert df_edges.shape[0] == 16

        df_nodes = pd.DataFrame({
            'node_id':['a', 'b', 'c', 'e', 'd', 'f', 'g', 'h', 'j', 'i', 'z', 'l'],
            })

        assert df_nodes.shape[0] == 12

        g = Graph(data=df_edges,
                  nodes=df_nodes,
                  keys={"src": "src",
                        "dst": "dst",
                        "weight": "value",
                        "id": "node_id"})

        assert isinstance(g, Graph)

        spectral_clustering = SpectralClustering(3, mode='spark', max_iterations=8)

        spectral_clustering.fit(g)

        labels_ = spectral_clustering.labels_.toPandas()

        assert labels_.shape[0] == 12
        assert labels_.cluster[labels_["node_id"] == 'a'].values[0] == labels_.cluster[labels_["node_id"] == 'b'].values[0]
        assert labels_.cluster[labels_["node_id"] == 'a'].values[0] == labels_.cluster[labels_["node_id"] == 'c'].values[0]
        assert labels_.cluster[labels_["node_id"] == 'a'].values[0] == labels_.cluster[labels_["node_id"] == 'e'].values[0]

        assert spectral_clustering.modularity_ > 0
