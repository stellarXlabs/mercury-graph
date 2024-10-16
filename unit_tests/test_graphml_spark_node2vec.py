import os
import pytest

from conftest import cleanup, TEST_FOLDER, TEST_SAVE, PATH_CACHE_RW

from mercury.graph.core import Graph, SparkInterface
from mercury.graph.graphml import SparkNode2Vec


# Create common graph with dead end nodes for tests
@pytest.fixture(scope="module")
def sample_g_dead_end_nodes():

    df_edges = SparkInterface().spark.createDataFrame(
        [
            ("A", "B", 2),
            ("B", "C", 1),
            ("C", "B", 4),
            ("C", "D", 3),
        ],
        ["src", "dst", "weight"],
    )

    df_nodes = SparkInterface().spark.createDataFrame(
        [("A", "ARON"), ("B", "BILL"), ("C", "CLAIR"), ("D", "DANIEL")],
        [
            "id",
            "name",
        ],
    )

    keys = {"src": "src", "dst": "dst", "weight": "weight", "id": "id"}

    g = Graph(data=df_edges, nodes=df_nodes, keys=keys)

    return g


class TestSparkNode2Vec(object):

    def test_embedding_etc(self):
        """
        Tests method SparkNode2Vec.embedding
        """
        E = SparkNode2Vec(dimension=2)

        assert E.embedding() is None
        assert E.model() is None
        assert E.save("filepath") is None
        assert E.get_most_similar_nodes("NA") is None

    def test_fit(self, g_comtrade):
        """
        Tests method SparkNode2Vec.fit
        """
        E = SparkNode2Vec(dimension=2, sampling_ratio=0.5, num_epochs=2)
        assert type(str(E)) is str and len(str(E)) > 0
        assert type(repr(E)) is str and len(repr(E)) > 0

        len_str = len(str(E))
        E.fit(g_comtrade)

        len_str_fit = len(str(E))
        assert len_str_fit > len_str

        assert E.model() is not None
        assert E.get_most_similar_nodes(
            E.embedding().select("word").first()[0]
        ).columns == ["word", "similarity"]

    def test__save__load(self, g_comtrade):
        """
        Tests method SparkNode2Vec._load
        """

        os.makedirs(TEST_FOLDER)

        E = SparkNode2Vec(
            dimension=4, sampling_ratio=1, num_epochs=3, w2v_min_count=1, w2v_max_iter=2
        )
        assert E.embedding() is None
        assert E.model() is None

        E.fit(g_comtrade)
        assert E.embedding() is not None
        assert E.model() is not None

        E.save(TEST_SAVE)

        F = SparkNode2Vec(load_file=TEST_SAVE)

        assert F.embedding() is not None
        assert F.model() is not None

    def test_graph_dead_end_nodes(self, sample_g_dead_end_nodes):
        """
        Tests the bug where random walks that reach dead end nodes are eliminated
        """

        E = SparkNode2Vec(dimension=2, sampling_ratio=1, num_epochs=2, w2v_min_count=1)
        E.fit(sample_g_dead_end_nodes)
        assert E.embedding().count() <= 4

    def test_num_paths_per_node(self, sample_g_dead_end_nodes):
        """
        Tests the correct num_paths_per_node are drawn from graph
        """

        E = SparkNode2Vec(
            dimension=32,
            sampling_ratio=1,
            num_epochs=2,
            num_paths_per_node=1,
            w2v_min_count=1,
        )
        E.fit(sample_g_dead_end_nodes)
        assert E.paths_.count() == 3

        E = SparkNode2Vec(
            dimension=32,
            sampling_ratio=1,
            num_epochs=2,
            num_paths_per_node=3,
            w2v_min_count=1,
        )
        E.fit(sample_g_dead_end_nodes)
        assert E.paths_.count() == 9

    def test_persist_rw(self, g_comtrade):

        E1 = SparkNode2Vec(
            dimension=2,
            sampling_ratio=1,
            num_epochs=2,
            num_paths_per_node=2,
            path_cache=PATH_CACHE_RW,
            use_cached_rw=False,
            n_partitions_cache=10,
        )
        E1.fit(g_comtrade)
        # check paths have been written
        assert E1.paths_.count() == SparkInterface().read_parquet(PATH_CACHE_RW).count()

        # Now load persisted random walks
        E2 = SparkNode2Vec(
            dimension=2, path_cache=PATH_CACHE_RW, use_cached_rw=True, num_epochs=2
        )
        E2.fit(g_comtrade)
        assert E2.paths_.count() == SparkInterface().read_parquet(PATH_CACHE_RW).count()

        assert E2.paths_.count() == E1.paths_.count()
