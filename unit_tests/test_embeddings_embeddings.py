import pytest

from mercury.graph.embeddings import Embeddings


def test_embeddings():
	e = Embeddings(10)

	assert e is not None
