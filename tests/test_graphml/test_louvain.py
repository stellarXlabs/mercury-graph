# Import dependencies
import pytest
import pandas as pd
from pyspark.sql.functions import collect_set
from mercury.graph.graphml import LouvainCommunities
from mercury.graph.core import Graph


class TestLouvain(object):
    def test_instancing(object):
        """
        Tests instancing and __init__ of the class LouvainCommunities
        """

        louvain_clustering = LouvainCommunities()

        assert isinstance(louvain_clustering, LouvainCommunities)
        
        assert louvain_clustering.max_iter == 10  # Default value


    def test_fit(self, spark):
        """
        Tests method LouvainCommunities.fit
        """

        # Create graph from edges and nodes in pyspark DataFrames
        df_edges = spark.createDataFrame(
            data=[
                (1, 0, 1),
                (2, 1, 1),
                (3, 2, 1),
                (4, 3, 1),
                (5, 3, 1),
                (5, 4, 1),
                (7, 6, 1),
                (8, 6, 1)
            ],
            schema=['src', 'dst', 'weight']
        )
        df_nodes = spark.createDataFrame(pd.DataFrame({
            'node_id': ['0', '1', '2', '3', '4', '5', '6', '7', '8'],
        }))

        g = Graph(data=df_edges,
                  nodes=df_nodes,
                  keys={"src": "src",
                        "dst": "dst",
                        "weight": "value",
                        "id": "node_id"})
        
        louvain_clustering = LouvainCommunities()

        # Get louvain partitions
        louvain_clustering.fit(g)
        communities = louvain_clustering.labels_

        ## Check communities' type (pyspark df)
        assert type(communities) == type(df_edges)

        ## Check if `communities` returned all nodes
        assert communities.count() == 9

        ## Test if the global optimum can be found
        
        # Get last pass
        col = communities.columns[-1]

        # Group by col and get nodes in each community
        partition = (
            communities
            .select('id', col)
            .groupBy(col)
            .agg(collect_set('id').alias('c'))
        ).collect()

        # Turn partition to list of sets
        partition = [set(row['c']) for row in partition]

        # Best communities (global maximum)
        c1, c2, c3 = {0, 1, 2}, {3, 4, 5}, {6, 7, 8}

        # Check if partition matches global max
        assert (c1 in partition) and (c2 in partition) and (c3 in partition)


    def test_sortPasses_lt10(self, spark):
        """
        Test sortPasses with maxPass < 10
        """

        # Define list of columns sorted by their integer part
        _passes = 5
        cols_expected = ['id'] + [f'pass{i}' for i in range(_passes + 1)]

        # Declare df with shuffled columns
        t = spark.createDataFrame(
            data=[tuple([1] * (_passes + 2))],
            schema=sorted(cols_expected)
        )

        louvain_clustering = LouvainCommunities()

        # Check if sortPasses sorts columns in expected order
        assert louvain_clustering._sort_passes(t) == cols_expected


    def test_sortPasses_gt10(self, spark):
        """
        Test sortPasses with maxPass > 10
        """

        # Define list of columns sorted by their integer part
        _passes = 30
        cols_expected = ['id'] + [f'pass{i}' for i in range(_passes + 1)]

        # Declare df with shuffled columns
        t = spark.createDataFrame(
            data=[tuple([1] * (_passes + 2))],
            schema=sorted(cols_expected)
        )
        
        louvain_clustering = LouvainCommunities()

        # Check if sortPasses sorts columns in expected order
        assert louvain_clustering._sort_passes(t) == cols_expected


    def test_missing_src(self, spark):
        """
        Test error raised by omitting src
        """

        t = spark.createDataFrame(data=[(1,)], schema=['dst'])

        expected_msg = "Input data is missing expected column 'src'."

        louvain_clustering = LouvainCommunities()

        with pytest.raises(ValueError, match=expected_msg):
            louvain_clustering._verify_data(df=t, 
                                            expected_cols_grouping=['src'],
                                            expected_cols_others=[])


    def test_missing_weight(self, spark):
        """
        Test if function assigns weight column correctly
        """

        t = spark.createDataFrame(data=[(1, 0)], schema=['src', 'dst'])

        expected_msg = "Input data is missing expected column 'weight'."

        louvain_clustering = LouvainCommunities()

        with pytest.raises(ValueError, match=expected_msg):
            assert louvain_clustering._verify_data(df=t,
                                                   expected_cols_grouping=['src', 'dst'],
                                                   expected_cols_others=['weight'])


    def test_lastPass_lt10(self, spark):
        """
        Test if lastPass returns last pass with <10 passes
        """

        _passes = 5

        t = spark.createDataFrame(
            data=[tuple([1] * (_passes + 2))],
            schema=['id'] + [f'pass{str(i)}' for i in range(_passes + 1)]
        )

        louvain_clustering = LouvainCommunities()

        assert louvain_clustering._last_pass(t) == f'pass{_passes}'


    def test_lastPass_gt10(self, spark):
        """
        Test if lastPass returns last pass with >10 passes
        """

        _passes = 30

        t = spark.createDataFrame(
            data=[tuple([1] * (_passes + 2))],
            schema=['id'] + [f'pass{str(i)}' for i in range(_passes + 1)]
        )

        louvain_clustering = LouvainCommunities()

        assert louvain_clustering._last_pass(t) == f'pass{_passes}'
