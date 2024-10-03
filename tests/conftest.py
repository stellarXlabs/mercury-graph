import logging
import shutil

import pytest

from mercury.graph.utils import is_spark_available


if is_spark_available():
    from mercury.graph.core.spark_interface import SparkInterface
    
    import pandas as pd
    pd.DataFrame.iteritems = pd.DataFrame.items  # In case pandas>=2.0 and Spark<3.4

    path_tmp_data = "./tmp_data"
    path_tmp_binf = "./tmp_binf"

    @pytest.fixture(scope="session")
    def spark(request):
        """ fixture for creating a spark session
            :param request: pytest.FixtureRequest object
            :return SparkSession for testing
        """

        logger = logging.getLogger("py4j")
        logger.setLevel(logging.FATAL)
        
        spark = SparkInterface().spark
        spark.sparkContext.setCheckpointDir("./tmp_checkpoint")
        yield spark
        
        request.addfinalizer(lambda: spark.stop())
        
        shutil.rmtree("./tmp_checkpoint")
