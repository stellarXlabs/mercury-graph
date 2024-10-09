import pytest

from mercury.graph.core.spark_interface import SparkInterface


def test_spark_interface():
	spark = SparkInterface()
	assert spark is not None
