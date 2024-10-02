import networkx
from mercury.graph.graphml import Transition


class TestTransition(object):
    def test_instancing(object):
        """
        Tests instancing of the class Transition
        """
        T = Transition()

        T.set_row(
            "Madrid", {"Bilbao": 400, "Barcelona": 700, "Sevilla": 500}
        )

        assert isinstance(T.G, networkx.classes.digraph.DiGraph)


    def test_fit(self):
        """
        Tests method Transition.fit
        """
        T = Transition()

        T.set_row("Madrid", {"Bilbao": 50, "Barcelona": 50})
        T.set_row("Bilbao", {"Santander": 25, "Algorta": 75})
        T.set_row("Barcelona", {"Reus": 25, "Andorra": 75})

        tm = networkx.adjacency_matrix(T.G, weight="weight")

        im = list(T.G.nodes).index("Madrid")
        ib = list(T.G.nodes).index("Bilbao")
        ia = list(T.G.nodes).index("Andorra")

        assert tm[[im], :].sum() == 100 and tm[[ib], :].sum() == 100 and tm[[ia], :].sum() == 0

        T.fit()

        tm = networkx.adjacency_matrix(T.G, weight="weight")

        for i in range(tm.shape[0]):
            assert tm[[i],:].sum() == 1

        tm = T.to_pandas()

        assert tm.loc["Madrid", "Bilbao"] == 0.5 and tm.loc["Algorta", "Algorta"] == 1

    def test_set_row(self):
        """
        Tests method Transition.set_row
        """
        T = Transition()

        T.set_row(
            "Madrid", {"Bilbao": 400, "Barcelona": 700, "Sevilla": 500}
        )
        assert T.G.number_of_nodes() == 4 and T.G.number_of_edges() == 3

        T.set_row("Bilbao", {"Santander": 100, "Algorta": 15})
        assert T.G.number_of_nodes() == 6 and T.G.number_of_edges() == 5

        T.set_row("Barcelona", {"Reus": 20})
        assert T.G.number_of_nodes() == 7 and T.G.number_of_edges() == 6

        T.set_row("Madrid", {"Bilbao": 400, "Barcelona": 700})
        assert T.G.number_of_nodes() == 7 and T.G.number_of_edges() == 5

        T.set_row("Sevilla", {"Madrid": 500, "Cádiz": 100})
        assert T.G.number_of_nodes() == 8 and T.G.number_of_edges() == 7

        T.set_row("Oviedo", {"Santander": 100})
        assert T.G.number_of_nodes() == 9 and T.G.number_of_edges() == 8

        T.set_row("Madrid", {"Barcelona": 700})
        assert T.G.number_of_nodes() == 9 and T.G.number_of_edges() == 7

        T.set_row("Barcelona", {"Reus": 20, "Bilbao": 700})
        assert T.G.number_of_nodes() == 9 and T.G.number_of_edges() == 8

        tm = T.to_pandas()

        assert tm.loc["Barcelona", "Reus"] == 20 and tm.loc["Barcelona", "Sevilla"] == 0
        assert (
            tm.loc["Oviedo", "Santander"] == 100 and tm.loc["Santander", "Oviedo"] == 0
        )
        assert tm.loc["Madrid", "Barcelona"] == 700 and tm.loc["Madrid", "Bilbao"] == 0

    def test_to_pandas(self):
        """
        Tests method Transition.to_pandas
        """
        T = Transition()

        T.set_row("Madrid", {"Bilbao": 50, "Barcelona": 50})
        T.set_row("Bilbao", {"Santander": 25, "Algorta": 75})
        T.set_row("Barcelona", {"Reus": 25, "Andorra": 75})

        tm = networkx.adjacency_matrix(T.G, weight="weight")

        T.fit()

        tm = T.to_pandas(num_iterations=0)

        assert tm.loc["Madrid", "Bilbao"] == 0 and tm.loc["Andorra", "Andorra"] == 1

        tm = T.to_pandas(num_iterations=1)

        assert tm["Madrid"].sum() == 0 and tm["Barcelona"].sum() == 0.5
        assert (
            tm.loc["Madrid", "Algorta"] == 0
            and tm.loc["Bilbao", "Algorta"] == 0.75
            and tm.loc["Reus", "Reus"] == 1
        )

        tm = T.to_pandas(num_iterations=2)

        assert (
            tm["Madrid"].sum() == 0
            and tm["Barcelona"].sum() == 0
            and tm["Bilbao"].sum() == 0
        )
        assert (
            tm.loc["Madrid", "Algorta"] == 0.375 and tm.loc["Madrid", "Reus"] == 0.125
        )

        tm = T.to_pandas(num_iterations=2000)

        assert (
            tm["Madrid"].sum() == 0
            and tm["Barcelona"].sum() == 0
            and tm["Bilbao"].sum() == 0
        )
        assert (
            tm.loc["Madrid", "Algorta"] == 0.375 and tm.loc["Madrid", "Reus"] == 0.125
        )
