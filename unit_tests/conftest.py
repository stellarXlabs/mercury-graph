import shutil
import pytest

import pandas as pd
pd.DataFrame.iteritems = pd.DataFrame.items  # In case pandas>=2.0 and Spark<3.4

from mercury.graph.core import Graph

TEST_FOLDER = "./spark_test_data"
TEST_SAVE = TEST_FOLDER + "/save"
PATH_CACHE_RW = TEST_FOLDER + "/cache"


def cleanup():
    shutil.rmtree(TEST_FOLDER, ignore_errors=True)


# Create common graph for tests from UN COMTRADE dataset
@pytest.fixture(scope="session")
def g_comtrade():
    df_edges = pd.read_csv("tutorials/data/un_comtrade_2016_sample.csv", sep="\t")

    df_names = pd.read_csv("tutorials/data/un_comtrade_2016_names.csv", sep="\t")

    df_edges = (
        df_edges.merge(df_names.rename(columns={"id": "ori"}), on="ori", how="inner")
        .drop(columns=["ori"])
        .rename(columns={"name": "ori"})
    )
    df_edges = (
        df_edges.merge(df_names.rename(columns={"id": "dest"}), on="dest", how="inner")
        .drop(columns=["dest"])
        .rename(columns={"name": "dest"})
    )

    df_nodes = df_edges[["dest"]]

    keys = {"src": "ori", "dst": "dest", "weight": "value", "id": "dest"}

    g = Graph(data=df_edges, nodes=df_nodes, keys=keys)

    return g
