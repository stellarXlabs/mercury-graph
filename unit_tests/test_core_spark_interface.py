import importlib, os, shutil

from unittest.mock import patch, MagicMock


mock_dgl = MagicMock()
mock_dgl.__spec__ = True


def test_spark_interface():
    with patch.dict('sys.modules', {'dgl': mock_dgl}):
        import mercury.graph.core.spark_interface as si

        importlib.reload(si)

        assert si.dgl_installed
        assert si.dgl == mock_dgl

        spark_int = si.SparkInterface(session = 'What?')

        assert si.SparkInterface._spark_session == 'What?'

        si.SparkInterface._spark_session = None

        assert isinstance(spark_int.dgl, MagicMock)

    import mercury.graph.core.spark_interface as si

    importlib.reload(si)

    spark_int = si.SparkInterface()

    data    = [('Alice', 34, 1), ('Bob', 45, 2), ('Charlie', 23, 3), ('Diana', 67, 4), ('Esther', 29, 4)]
    columns = ['Name', 'Age', 'ID']
    df = spark_int.spark.createDataFrame(data, columns)

    test_path = 'spark_int_test'
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    csv_path     = '%s/data.csv' % test_path
    json_path    = '%s/data.json' % test_path
    parquet_path = '%s/data.parquet' % test_path

    df.write.csv(csv_path, header = True, mode = 'overwrite')
    df.write.json(json_path, mode = 'overwrite')
    df.write.parquet(parquet_path, mode = 'overwrite')

    df2 = spark_int.read_csv(csv_path, header = True)
    assert df2.count() == 5
    cols = df2.columns
    cols.sort()
    assert cols == ['Age', 'ID', 'Name']

    df2 = spark_int.read_json(json_path)
    assert df2.count() == 5
    cols = df2.columns
    cols.sort()
    assert cols == ['Age', 'ID', 'Name']

    df2 = spark_int.read_parquet(parquet_path)
    assert df2.count() == 5
    cols = df2.columns
    cols.sort()
    assert cols == ['Age', 'ID', 'Name']

    df2 = spark_int.read_text(csv_path)
    assert len(df2.columns) == 1

    df2 = spark_int.read(parquet_path, format = 'parquet')
    assert df2.count() == 5
    cols = df2.columns
    cols.sort()
    assert cols == ['Age', 'ID', 'Name']

    df.createOrReplaceTempView('people')
    sql_df = spark_int.sql('SELECT * FROM people WHERE Age > 40')
    assert sql_df.count() == 2
    assert set([row['Name'] for row in sql_df.collect()]) == {'Bob', 'Diana'}

    def multiply_by_two(x):
        return 2*x

    assert multiply_by_two(3) == 6

    spark_int.udf(multiply_by_two, returnType = 'int')

    udf_df = spark_int.sql('SELECT Name, multiply_by_two(Age) as Age_x2 FROM people')
    assert udf_df.count() == 5
    assert udf_df.columns == ['Name', 'Age_x2']

    shutil.rmtree(test_path)


# test_spark_interface()
