import pytest
import numpy as np

from mercury.graph.embeddings import Embeddings


class TestEmbeddings(object):
    def test_instancing(object):
        """
        Tests instancing of the class Embeddings
        """

        with pytest.raises(TypeError):
            e = Embeddings()

        e = Embeddings(dimension=4)
        assert isinstance(e, Embeddings)

    def test___init__(self):
        """
        Tests method Embeddings.__init__
        """
        e = Embeddings(dimension=4)
        assert isinstance(e, Embeddings)
        assert e.dimension == 4
        assert not hasattr(e, "embeddings_matrix_")

        e = Embeddings(dimension=11, num_elements=50, mean=1, sd=2)
        assert isinstance(e, Embeddings)
        assert e.dimension == 11
        assert e.num_elements == 50
        assert e.mean == 1
        assert e.sd == 2
        assert isinstance(e.embeddings_matrix_, np.ndarray)

    def test_as_numpy(self):
        """
        Tests method Embeddings.as_numpy
        """
        e = Embeddings(dimension=4)
        assert e.as_numpy() is None

        e = Embeddings(dimension=11, num_elements=50, mean=1, sd=2)
        m = e.as_numpy()
        assert isinstance(m, np.ndarray)
        assert np.size(m, 0) == 50
        assert np.size(m, 1) == 11

        with pytest.raises(TypeError):
            m = e.as_numpy(2)

    def test_fit(self):
        """
        Tests method Embeddings.fit
        """
        e = Embeddings(dimension=30, num_elements=15)
        e_bid = Embeddings(dimension=30, num_elements=15, bidirectional=True)
        # Same random initial embedding matrix for comparison
        e_bid.embeddings_matrix_ = e.embeddings_matrix_

        def get_changes(m1, m2):
            changes = []

            for i in range(m1.shape[0]):
                changed = False

                for j in range(m1.shape[1]):
                    if m1[i, j] != m2[i, j]:
                        changed = True
                        break

                if changed:
                    changes.append(i)

            return changes

        emb_matrix = e.as_numpy().copy()

        assert get_changes(e.as_numpy().copy(), emb_matrix) == []

        converge = np.array([[0, 1]])
        e.fit(converge=converge)
        assert get_changes(e.as_numpy().copy(), emb_matrix) == [0]

        emb_matrix = e.as_numpy().copy()

        converge = np.array([[4, 5], [4, 5], [9, 10]])
        e.fit(converge=converge)
        assert get_changes(e.as_numpy().copy(), emb_matrix) == [4, 9]

        emb_matrix = e.as_numpy().copy()

        converge = np.array([[4, 5], [4, 5], [9, 10]])
        e_bid.fit(converge=converge)
        assert get_changes(e_bid.as_numpy().copy(), emb_matrix) == [4, 5, 9, 10]

        emb_matrix = e.as_numpy().copy()

        diverge = np.array([[7, 8]])
        e.fit(diverge=diverge)
        assert get_changes(e.as_numpy().copy(), emb_matrix) == [7]

        emb_matrix = e.as_numpy().copy()

        e.fit(converge=converge, diverge=diverge)
        assert get_changes(e.as_numpy().copy(), emb_matrix) == [4, 7, 9]

        emb_matrix = e.as_numpy().copy()

        e_bid.fit(converge=converge, diverge=diverge)
        assert get_changes(e_bid.as_numpy().copy(), emb_matrix) == [4, 5, 7, 8, 9, 10]

    def test_get_most_similar_embeddings(self):
        """
        Tests method Embeddings.get_most_similar_embeddings
        """
        # Set embedding matrix to known most similar embeddings
        e = Embeddings(dimension=2)
        e.embeddings_matrix_ = np.array([[1, 0], [0, 1], [1, 0.5], [-1, 0]])

        similar_nodes, similarities = e.get_most_similar_embeddings(
            0, metric="cosine", k=10
        )
        assert list(similar_nodes) == [2, 1, 3]
        assert round(similarities[0], 4) == 0.8944
        assert similarities[1] == 0
        assert similarities[2] == -1

        similar_nodes, similarities = e.get_most_similar_embeddings(
            0, metric="euclidean", k=10
        )
        assert list(similar_nodes) == [2, 1, 3]
        assert round(similarities[0], 4) == 0.6667
        assert round(similarities[1], 4) == 0.4142
        assert round(similarities[2], 4) == 0.3333

        with pytest.raises(ValueError):
            similar_nodes, similarities = e.get_most_similar_embeddings(
                0, metric="non_existent", k=10
            )
