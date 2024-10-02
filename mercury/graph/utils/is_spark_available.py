import os


def is_spark_available():
    """
    Tests if Apache Spark is available. `mercury.graph` can be used platform independently. If `Spark` is available, `mercury.graph` assumes
    [graphframes](http://graphframes.github.io/graphframes/docs/_site/index.html) is also installed and can appropriately
    link to `Spark`.

    If is_spark_available() returns `False`, you can still use `mercury.graph` with [networkx](https://networkx.github.io/) and,
    possibly, other technologies.

    Technically, it just checks if the environment variable SPARK_HOME is set.
    """

    return os.environ.get("SPARK_HOME") is not None