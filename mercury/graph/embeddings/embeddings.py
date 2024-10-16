import numpy as np

from scipy.spatial.distance import cdist
from mercury.graph.core._njit import njit

from mercury.graph.core.base import BaseClass


@njit
def _elliptic_rotate(self_em, iu, iv, cos_w, sin_w):
    for i in range(len(iu)):
        j = iu[i]

        u = self_em[j]
        v = self_em[iv[i]]

        sc = np.dot(u, v) / np.dot(u, u)
        pv = sc * u
        tv = v - pv

        self_em[j] = cos_w * pv / sc + sin_w * tv

    return self_em


class Embeddings(BaseClass):
    """
    This class holds a matrix object that is interpreted as the embeddings for any list of objects, not only the nodes of a graph. You
    can see this class as the internal object holding the embedding for other classes such as class GraphEmbedding.

    Args:
        dimension (int): The number of columns in the embedding. See note below.
        num_elements (int): The number of rows in the embedding. You can leave this empty on creation and then use initialize_as() to
            automatically match the nodes in a graph.
        mean (float): The (expected) mean of the initial values.
        sd (float): The (expected) standard deviation of the initial values.
        learn_step (float): The size of the learning step elements get approached or moved away. Units are hexadecimal degrees in along
            an ellipse.
        bidirectional (bool): Should the changes apply only to the elements of first column (False) or to both.

    Note:
        **On dimension:** Embeddings cannot be zero (that is against the whole concept). Smaller dimension embeddings can only hold
        few elements without introducing spurious correlations by some form of 'birthday attack' phenomenon as elements increase. Later
        it is very hard to get rid of that spurious 'knowledge'. **Solution**: With may elements, you have to go to high enough dimension
        even if the structure is simple. Pretending to fit many embeddings in low dimension without them being correlated is like
        pretending to plot a trillion random points in a square centimeter while keeping them 1 mm apart from each other: It's simply
        impossible!
    """

    def __init__(
        self, dimension, num_elements=0, mean=0, sd=1, learn_step=3, bidirectional=False
    ):
        self.dimension = dimension
        self.num_elements = num_elements
        self.mean = mean
        self.sd = sd
        self.learn_step = learn_step
        self.bidirectional = bidirectional

        if self.num_elements > 0:
            self.embeddings_matrix_ = np.random.normal(
                self.mean, self.sd, (self.num_elements, self.dimension)
            )

    def fit(self, converge=None, diverge=None):
        """
        Apply a learning step to the embedding.

        Args:
            converge (numpy matrix of two columns): A matrix of indices to elements meaning (first column) should be approached to
                (second column).
            diverge (numpy matrix of two columns): A matrix of indices to elements meaning (first column) should be moved away from
                (second column).

        Returns:
            (self): Fitted self (or raises an error)

        Note:
            Embeddings start being randomly distributed and hold no structure other than spurious correlations. Each time you apply a
            learning step by calling this method, you are tweaking the embedding to approach some rows and/or move others away. You can use
            both converge and diverge or just one of them and call this as many times you want with varying learning step. A proxy of how
            much an embedding can learn can be estimated by measuring how row correlations are converging towards some asymptotic values.
        """

        w = self.learn_step * np.pi / 180

        cos_w = np.cos(w)
        sin_w = np.sin(w)

        if converge is not None:
            self.embeddings_matrix_ = _elliptic_rotate(
                self.embeddings_matrix_, converge[:, 0], converge[:, 1], cos_w, sin_w
            )

            if self.bidirectional:
                self.embeddings_matrix_ = _elliptic_rotate(
                    self.embeddings_matrix_,
                    converge[:, 1],
                    converge[:, 0],
                    cos_w,
                    sin_w,
                )

        if diverge is not None:
            self.embeddings_matrix_ = _elliptic_rotate(
                self.embeddings_matrix_, diverge[:, 0], diverge[:, 1], cos_w, -sin_w
            )

            if self.bidirectional:
                self.embeddings_matrix_ = _elliptic_rotate(
                    self.embeddings_matrix_, diverge[:, 1], diverge[:, 0], cos_w, -sin_w
                )

        return self

    def as_numpy(self):
        """
        Return the embedding as a numpy matrix where each row is an embedding.
        """
        if not hasattr(self, "embeddings_matrix_"):
            return

        return self.embeddings_matrix_

    def get_most_similar_embeddings(self, index, k=5, metric="cosine"):
        """
        Given an index of a vector in the embedding matrix, returns the k most similar embeddings in the matrix

        Args:
            index (int): index of the vector in the matrix that we want to compute the similar embeddings
            k (int): Number of most similar embeddings to return
            metric (str): metric to use as a similarity.

        Returns:
            (list): list of k most similar nodes as indices and list of similarities of the most similar nodes
        """
        if metric == "cosine":
            similarities = (
                1
                - cdist(
                    np.expand_dims(self.as_numpy()[index], axis=0),
                    self.as_numpy(),
                    "cosine",
                )[0]
            )

        elif metric == "euclidean":
            similarities = 1 / (
                1
                + cdist(
                    np.expand_dims(self.as_numpy()[index], axis=0),
                    self.as_numpy(),
                    "euclidean",
                )[0]
            )

        else:
            raise ValueError("Unknown Distance Metric: %s" % metric)

        ordered_indices = np.argsort(similarities)[::-1][1 : (k + 1)]
        ordered_similarities = similarities[ordered_indices]

        return ordered_indices, ordered_similarities
