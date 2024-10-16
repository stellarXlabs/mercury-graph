from mercury.graph.graphml import SparkRandomWalker


class TestSparkRandomWalker(object):
    def test_instancing(object):
        """
        Tests instancing of the class SparkRandomWalker
        """
        E = SparkRandomWalker()
        assert isinstance(E, SparkRandomWalker)

        assert type(str(E)) is str and len(str(E)) > 0
        assert type(repr(E)) is str and len(repr(E)) > 0

    def test_random_walks(self, g_comtrade):
        """
        Tests method SparkRandomWalker.random_walks
        """
        E = SparkRandomWalker(num_epochs=2, batch_size=1)
        len_str = len(str(E))
        E.fit(g_comtrade, source_id="Algeria")
        walks1 = E.paths_

        len_str_fit = len(str(E))
        assert len_str_fit > len_str

        # Sampling edges produces fewer walks
        E = SparkRandomWalker(num_epochs=2, batch_size=1, n_sampling_edges=2)
        E.fit(g_comtrade, source_id="Algeria")
        walks2 = E.paths_

        assert walks1.count() > walks2.count()
