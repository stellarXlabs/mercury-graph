# mercury-graph

**`mercury-graph`** is a Python library that offers **graph analytics capabilities with a technology-agnostic API**, allowing users to use a curated range of performant and scalable algorithms and utilities regardless of the technologies employed (pure Python, [Numba](https://numba.pydata.org/)-compiled, [**networkx**](https://networkx.org/), distributed Spark [**graphframes**](https://graphframes.github.io/graphframes/docs/_site/index.html), etc.).

Currently implemented **submodules** in `mercury.graph` include:

- [**`mercury.graph.core`**](reference/core.md), with the main classes of the library that create and store the graphs' data and properties.

- [**`mercury.graph.ml`**](reference/ml.md), with graph theory and machine learning algorithms such as Louvain community detection, spectral clustering, Markov chains, spreading activation-based diffusion models and graph random walkers.

- [**`mercury.graph.embeddings`**](reference/embeddings.md), with classes that calculate graph embeddings in different ways, such as following the Node2Vec algorithm.

- [**`mercury.graph.viz`**](reference/viz.md), with capabilities for graph visualization.


## Python installation

The easiest way to install `mercury-graph` is using `pip`:

```bash
    pip install mercury-graph
```

### Repository

Website: [https://github.com/BBVA/mercury-graph](https://github.com/BBVA/mercury-graph)


## Help and support

It is a part of [**`mercury`**](https://www.bbvaaifactory.com/mercury/), a collaborative library developed by the **Advanced Analytics community at BBVA** that offers a broad range of tools to simplify and accelerate data science workflows. This library was originally an Inner Source project, but some components, like `mercury.graph`, have been released as Open Source.

  * [Mercury team](mailto:mercury.group@bbva.com?subject=[mercury-graph])
  * [Issues](https://github.com/BBVA/mercury-graph/issues)