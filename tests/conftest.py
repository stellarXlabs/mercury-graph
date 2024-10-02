import logging
import shutil

import pytest

from mercury.graph.utils import is_spark_available

if is_spark_available():
    from pyspark.sql import SparkSession
    
    import pandas as pd
    pd.DataFrame.iteritems = pd.DataFrame.items

    path_tmp_data = "./tmp_data"
    path_tmp_binf = "./tmp_binf"

    @pytest.fixture(scope="session")
    def spark(request):
        """ fixture for creating a spark session
            :param request: pytest.FixtureRequest object
            :return SparkSession for testing
        """
        ss = (
            SparkSession.builder.appName("mercury.graph")
            .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.0-s_2.12")
            .getOrCreate()
        )
        ss.sparkContext.setCheckpointDir("./tmp_checkpoint")
        logger = logging.getLogger("py4j")
        logger.setLevel(logging.FATAL)
        yield ss
        request.addfinalizer(lambda: ss.stop())
        shutil.rmtree("./tmp_checkpoint")
