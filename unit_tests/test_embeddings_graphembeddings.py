import os
import bz2
import pickle
import pytest
import numpy as np
import pandas as pd
import scipy

from mercury.graph.core import Graph

from mercury.graph.embeddings import GraphEmbedding, Embeddings
from mercury.graph.embeddings.graphembeddings import (
    _random_node_weighted,
    _random_walks,
)


# Create small common graph for GraphEmbeddings tests
@pytest.fixture(scope="module")
def sample_g():

    df_edges = pd.DataFrame(
        {
            "src": ["A", "B", "C", "D"],
            "dst": ["C", "D", "A", "B"],
            "num_times": [780, 670, 893, 890],
        }
    )

    df_nodes = df_edges[["src"]].rename(columns={"src": "node_id"})

    keys = {"src": "src", "dst": "dst", "weight": "num_times", "id": "node_id"}

    g = Graph(data=df_edges, nodes=df_nodes, keys=keys)

    return g


class TestGraphEmbedding(object):
    def test_instancing(object):
        """
        Tests instancing of the class GraphEmbedding
        """
        ge = GraphEmbedding(dimension=30)

        assert isinstance(ge, GraphEmbedding)
        assert ge.dimension == 30

    def test___getitem__(self, sample_g):
        """
        Tests method GraphEmbedding.__getitem__
        """
        ge = GraphEmbedding(dimension=30, n_jumps=100, learn_step=3, bidirectional=True)
        ge.fit(sample_g)

        with pytest.raises(ValueError):
            ge["wrong_access"]

        assert isinstance(ge["A"], np.ndarray)

    def test___init__(self, sample_g):
        """
        Tests method GraphEmbedding.__init__
        """
        ge = GraphEmbedding(dimension=30, n_jumps=100, learn_step=3, bidirectional=True)
        ge.fit(sample_g)

        assert ge.node_ids == list(sample_g.networkx.nodes)
        assert isinstance(ge.graph_embedding_, Embeddings)

    def test__load(self, sample_g, manage_path_tmp_binf):
        """
        Tests method GraphEmbedding._load
        """
        self.test_save(sample_g, manage_path_tmp_binf)

        fn1 = manage_path_tmp_binf + "/test_gremb_with.graph"
        fn2 = manage_path_tmp_binf + "/test_gremb_without.graph"

        ge1 = GraphEmbedding(load_file=fn1)
        ge2 = GraphEmbedding(load_file=fn2)

        assert isinstance(ge1, GraphEmbedding)
        assert isinstance(ge2, GraphEmbedding)

        assert ge1.node_ids == ge2.node_ids
        assert ge1.TotW == ge2.TotW
        assert (
            sum(
                sum(
                    ge1.graph_embedding_.embeddings_matrix_
                    != ge2.graph_embedding_.embeddings_matrix_
                )
            )
            > 0
        )

        fn3 = manage_path_tmp_binf + "/test_gremb_wrong_head.graph"
        with bz2.BZ2File(fn3, "w") as f:
            pickle.dump("WRONG_HEAD", f)
        with pytest.raises(ValueError):
            ge1._load(fn3)

    def test_embedding(self, sample_g):
        """
        Tests method GraphEmbedding.embedding
        """
        ge = GraphEmbedding(dimension=30, n_jumps=100, learn_step=3, bidirectional=True)
        ge.fit(sample_g)
        assert isinstance(ge.embedding(), Embeddings)

    def test_get_most_similar_nodes(self):
        """
        Tests method GraphEmbedding.get_most_similar_nodes
        """
        ge = GraphEmbedding(dimension=2)
        # Set embedding matrix to known most similar embeddings
        ge.node_ids = ["A", "B", "C", "D"]
        ge.graph_embedding_ = Embeddings(dimension=2)
        ge.graph_embedding_.embeddings_matrix_ = np.array(
            [[1, 0], [0, 1], [1, 0.5], [-1, 0]]
        )

        # cosine distance
        similar_nodes_df = ge.get_most_similar_nodes("A", metric="cosine", k=10)
        assert list(similar_nodes_df["word"]) == ["C", "B", "D"]
        assert round(similar_nodes_df.loc[0, "similarity"], 4) == 0.8944
        assert similar_nodes_df.loc[1, "similarity"] == 0
        assert similar_nodes_df.loc[2, "similarity"] == -1

        # euclidean distance
        similar_nodes_df = ge.get_most_similar_nodes("A", metric="euclidean", k=10)
        assert list(similar_nodes_df["word"]) == ["C", "B", "D"]
        assert round(similar_nodes_df.loc[0, "similarity"], 4) == 0.6667
        assert round(similar_nodes_df.loc[1, "similarity"], 4) == 0.4142
        assert round(similar_nodes_df.loc[2, "similarity"], 4) == 0.3333

        # returning indices
        similar_nodes_df = ge.get_most_similar_nodes(
            "A", metric="cosine", k=10, return_as_indices=True
        )
        assert list(similar_nodes_df["word"]) == [2, 1, 3]
        assert round(similar_nodes_df.loc[0, "similarity"], 4) == 0.8944
        assert similar_nodes_df.loc[1, "similarity"] == 0
        assert similar_nodes_df.loc[2, "similarity"] == -1

    def test_numba_functions(self):
        """
        Tests functions _random_node_weighted and _random_walks
        """
        j_matrix = scipy.sparse.csr_matrix((7, 7))

        j_matrix[0, 1] = 25
        j_matrix[0, 2] = 10
        j_matrix[0, 3] = 5

        j_matrix[1, 0] = 1
        j_matrix[1, 2] = 2

        j_matrix[2, 1] = 1
        j_matrix[2, 3] = 3

        j_matrix[3, 2] = 2

        j_matrix[4, 5] = 2
        j_matrix[4, 6] = 3

        j_matrix[5, 4] = 1
        j_matrix[5, 6] = 9

        N = j_matrix.shape[1]
        M = j_matrix.getnnz()

        assert N == 7 and M == 12

        r_ini = np.zeros(N, dtype=int)
        r_len = np.zeros(N, dtype=int)
        r_sum = np.zeros(N, dtype=float)
        r_col = np.zeros(M, dtype=int)
        r_wgt = np.zeros(M, dtype=float)

        i = 0
        for r in range(N):
            r_ini[r] = i

            i_col = j_matrix[r].nonzero()[1]
            L = len(i_col)

            r_len[r] = L

            for k in range(L):
                c = i_col[k]
                w = j_matrix[r, c]

                r_sum[r] += w
                r_col[i] = c
                r_wgt[i] = w

                i += 1

        TotW = sum(r_sum)

        assert (
            r_ini[0] == 0
            and r_ini[1] == 3
            and r_ini[2] == 5
            and r_ini[3] == 7
            and r_ini[4] == 8
            and r_ini[5] == 10
            and r_ini[6] == 12
        )
        assert (
            r_len[0] == 3
            and r_len[1] == 2
            and r_len[2] == 2
            and r_len[3] == 1
            and r_len[4] == 2
            and r_len[5] == 2
            and r_len[6] == 0
        )
        assert (
            r_sum[0] == 40
            and r_sum[1] == 3
            and r_sum[2] == 4
            and r_sum[3] == 2
            and r_sum[4] == 5
            and r_sum[5] == 10
            and r_sum[6] == 0
        )

        assert (
            r_col[0] == 1
            and r_col[1] == 2
            and r_col[2] == 3
            and r_col[3] == 0
            and r_col[4] == 2
            and r_col[5] == 1
        )
        assert (
            r_col[6] == 3
            and r_col[7] == 2
            and r_col[8] == 5
            and r_col[9] == 6
            and r_col[10] == 4
            and r_col[11] == 6
        )

        assert (
            r_wgt[0] == 25
            and r_wgt[1] == 10
            and r_wgt[2] == 5
            and r_wgt[3] == 1
            and r_wgt[4] == 2
            and r_wgt[5] == 1
        )
        assert (
            r_wgt[6] == 3
            and r_wgt[7] == 2
            and r_wgt[8] == 2
            and r_wgt[9] == 3
            and r_wgt[10] == 1
            and r_wgt[11] == 9
        )

        assert TotW == 64

        i_sum = np.zeros(N, dtype=int)

        for _ in range(1000):
            i = _random_node_weighted(r_sum, TotW)
            assert i >= 0 and i < N
            i_sum[i] += 1

        assert sum(i_sum) == 1000 and i_sum[6] == 0

        # Symmetric CI for binomial proportion with Confidence level > 99.99999%

        assert i_sum[0] > 541 and i_sum[0] < 705
        assert i_sum[2] > 29 and i_sum[2] < 114
        assert i_sum[5] > 101 and i_sum[5] < 225

        (converge, diverge) = _random_walks(
            r_ini, r_len, r_sum, r_col, r_wgt, TotW, 100, 100
        )

        assert (
            converge.dtype == "int32" or converge.dtype == "int64"
        ) and diverge.dtype == converge.dtype
        assert converge.shape == (100, 2) and diverge.shape == (100, 2)

        assert min(converge[:, 0]) <= 1 and max(converge[:, 0]) < 7
        assert min(converge[:, 1]) <= 1 and max(converge[:, 1]) < 7

        assert min(diverge[:, 0]) <= 1 and max(diverge[:, 0]) < 7
        assert min(diverge[:, 1]) <= 1 and max(diverge[:, 1]) < 7

        assert any(converge[:, 0] - converge[:, 1] != 0) and any(
            diverge[:, 0] - diverge[:, 1] != 0
        )

        (converge, diverge) = _random_walks(
            r_ini, r_len, r_sum, r_col, r_wgt, TotW, 2000, 10
        )

        assert converge.shape == (2000, 2) and diverge.shape == (2000, 2)

        assert min(converge[:, 0]) <= 1 and max(converge[:, 0]) == 5
        assert min(converge[:, 1]) <= 1 and max(converge[:, 1]) == 6

        assert min(diverge[:, 0]) <= 1 and max(diverge[:, 0]) == 5
        assert min(diverge[:, 1]) <= 1 and max(diverge[:, 1]) == 6

        assert any(converge[:, 0] - converge[:, 1] != 0) and any(
            diverge[:, 0] - diverge[:, 1] != 0
        )

    def test_fit(self, sample_g):
        """
        Tests method GraphEmbedding.random_walks
        """
        ge = GraphEmbedding(dimension=30, n_jumps=100, learn_step=3, bidirectional=True)
        assert ge.embedding() is None
        ge.fit(sample_g)
        assert ge.embedding().as_numpy() is not None

    def test_save(self, sample_g, manage_path_tmp_binf):
        """
        Tests method GraphEmbedding.save
        """
        fn1 = manage_path_tmp_binf + "/test_gremb_with.graph"
        fn2 = manage_path_tmp_binf + "/test_gremb_without.graph"

        ge = GraphEmbedding(dimension=30, n_jumps=100, learn_step=3, bidirectional=True)
        ge.fit(sample_g)

        ge.save(fn1, save_embedding=True)
        ge.save(fn2)
        assert os.path.getsize(fn1) > os.path.getsize(fn2)
