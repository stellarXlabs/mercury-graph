import pandas as pd
import networkx
from mercury.graph.graphml import Transition
from mercury.graph.core import Graph


df_edges = pd.DataFrame({
    'src': ['Madrid', 'Madrid', 'Bilbao', 'Bilbao', 'Barcelona', 'Barcelona'],
    'dst': ['Bilbao', 'Barcelona', 'Santander', 'Algorta', 'Reus', 'Andorra'],
    'value': [50, 50, 25, 75, 25, 75]
})

df_nodes = pd.DataFrame({
    'node_id': ['Madrid', 'Barcelona', 'Bilbao', 'Santander', 'Andorra', 'Reus', 'Algorta'],
})

g = Graph(data=df_edges,
          nodes=df_nodes,
          keys={"src": "src",
                "dst": "dst",
                "weight": "value",
                "id": "node_id"})


class TestTransition(object):

    def test_fit(self):
        """
        Tests method Transition.fit
        """
        T = Transition()

        # Adjacency matrix of original graph before Transition.fit()

        tm = networkx.adjacency_matrix(g.networkx, weight="weight")

        im = list(g.networkx.nodes).index("Madrid")
        ib = list(g.networkx.nodes).index("Bilbao")
        ia = list(g.networkx.nodes).index("Andorra")

        assert tm[[im], :].sum() == 100 and tm[[ib], :].sum() == 100 and tm[[ia], :].sum() == 0

        # Adjacency matrix of graph after Transition.fit()

        T.fit(g)

        tm = networkx.adjacency_matrix(T.G_markov_.networkx, weight="weight")

        for i in range(tm.shape[0]):
            assert tm[[i],:].sum() == 1

        tm = T.to_pandas()

        assert tm.loc["Madrid", "Bilbao"] == 0.5 and tm.loc["Algorta", "Algorta"] == 1


    def test_to_pandas(self):
        """
        Tests method Transition.to_pandas
        """
        T = Transition()

        T.fit(g)

        # 0 iterations

        tm = T.to_pandas(num_iterations=0)

        assert tm.loc["Madrid", "Bilbao"] == 0 and tm.loc["Andorra", "Andorra"] == 1

        # 1 iteration

        tm = T.to_pandas(num_iterations=1)

        assert tm["Madrid"].sum() == 0 and tm["Barcelona"].sum() == 0.5
        assert (
            tm.loc["Madrid", "Algorta"] == 0
            and tm.loc["Bilbao", "Algorta"] == 0.75
            and tm.loc["Reus", "Reus"] == 1
        )

        # 2 iterations

        tm = T.to_pandas(num_iterations=2)

        assert (
            tm["Madrid"].sum() == 0
            and tm["Barcelona"].sum() == 0
            and tm["Bilbao"].sum() == 0
        )
        assert (
            tm.loc["Madrid", "Algorta"] == 0.375 and tm.loc["Madrid", "Reus"] == 0.125
        )

        # 2000 iterations

        tm = T.to_pandas(num_iterations=2000)

        assert (
            tm["Madrid"].sum() == 0
            and tm["Barcelona"].sum() == 0
            and tm["Bilbao"].sum() == 0
        )
        assert (
            tm.loc["Madrid", "Algorta"] == 0.375 and tm.loc["Madrid", "Reus"] == 0.125
        )
