import importlib.util


find_spark = importlib.util.find_spec('pyspark')

if find_spark is None:
    pyspark_installed = False
    graphframes_installed = False
else:
    pyspark_installed = True

    from pyspark.sql import SparkSession

    find_graphframes = importlib.util.find_spec('graphframes')

    if find_graphframes is None:
        graphframes_installed = False
    else:
        graphframes_installed = True
        import graphframes as gf


# Define the Spark configuration options by default
default_spark_config = {
    'appName': 'LocalSparkApp',
    'master': 'local[*]',

    # Add other configurations as needed
    # 'spark.executor.memory': '2g',
    # 'spark.driver.memory': '1g',
}


class SparkInterface:
    """
    A class that provides an interface for interacting with Apache Spark.

    Attributes:
        _spark_session (pyspark.sql.SparkSession): The shared Spark session.
        _graphframes (module): The shared graphframes namespace.

    Methods:
        __init__(self, config=None): Initializes the SparkInterface object.
        _create_spark_session(config): Creates a Spark session.
        spark: Property that returns the shared Spark session.
        graphframes: Property that returns the shared graphframes namespace.
        read_csv(path, **kwargs): Reads a CSV file into a DataFrame.
        read_parquet(path, **kwargs): Reads a Parquet file into a DataFrame.
        read_json(path, **kwargs): Reads a JSON file into a DataFrame.
        read_text(path, **kwargs): Reads a text file into a DataFrame.
        read(path, format, **kwargs): Reads a file into a DataFrame.
        sql(query): Executes a SQL query.
        udf(f, returnType): Registers a user-defined function (UDF).
        stop(): Stops the Spark session.

    Args:
        config (dict, optional): A dictionary of Spark configuration options.
            If not provided, the configuration in the global variable `default_spark_config` will be used.
    """

    _spark_session = None   # Class variable to hold the shared Spark session
    _graphframes = None     # Class variable to hold the shared graphframes namespace


    def __init__(self, config=None):
        if SparkInterface._spark_session is None:
            SparkInterface._spark_session = self._create_spark_session(config)

        if SparkInterface._graphframes is None and graphframes_installed:
            SparkInterface._graphframes = gf


    @staticmethod
    def _create_spark_session(config):
        if config is None:
            config = default_spark_config

        spark_builder = SparkSession.builder

        for key, value in config.items():
            spark_builder = spark_builder.config(key, value)

        return spark_builder.getOrCreate()


    @property
    def spark(self):
        return SparkInterface._spark_session


    @property
    def graphframes(self):
        return SparkInterface._graphframes


    def read_csv(self, path, **kwargs):
        return self.spark.read.csv(path, **kwargs)


    def read_parquet(self, path, **kwargs):
        return self.spark.read.parquet(path, **kwargs)


    def read_json(self, path, **kwargs):
        return self.spark.read.json(path, **kwargs)


    def read_text(self, path, **kwargs):
        return self.spark.read.text(path, **kwargs)


    def read(self, path, format, **kwargs):
        return self.spark.read.format(format).load(path, **kwargs)


    def sql(self, query):
        return self.spark.sql(query)


    def udf(self, f, returnType):
        return self.spark.udf.register(f.__name__, f, returnType)


    def stop(self):
        return self.spark.stop()
