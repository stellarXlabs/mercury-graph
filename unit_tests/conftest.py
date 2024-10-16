import os
import shutil
import pytest

import pandas as pd

pd.DataFrame.iteritems = pd.DataFrame.items  # In case pandas>=2.0 and Spark<3.4

from mercury.graph.core import Graph

PATH_TMP_BINF = "./tmp_binf"
TEST_FOLDER = "./spark_test_data"
TEST_SAVE = TEST_FOLDER + "/save"
PATH_CACHE_RW = TEST_FOLDER + "/cache"


@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    # Clean testing directories once we are finished
    def remove_test_dirs():
        shutil.rmtree(TEST_FOLDER, ignore_errors=True)
        shutil.rmtree(".checkpoint", ignore_errors=True)

    request.addfinalizer(remove_test_dirs)


# Manage a temporary folder for testing binary files
@pytest.fixture(scope="session")
def manage_path_tmp_binf():
    if not os.path.exists(PATH_TMP_BINF):
        os.makedirs(PATH_TMP_BINF)

    yield PATH_TMP_BINF

    if os.path.exists(PATH_TMP_BINF):
        shutil.rmtree(PATH_TMP_BINF)


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
