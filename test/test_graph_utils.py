import tempfile
from pathlib import Path
from unittest import TestCase

import networkx as nx
from networkx.utils import graphs_equal
from qeco.utils.graph_utils import (
    degree_based_lower_bound,
    load_graph,
    node_edge_based_upper_bound,
    swap_strategy_simplify,
)
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes.routing import SwapStrategy


class TestGraphUtils(TestCase):
    def setUp(self):
        pass

    def test_degree_based_lower_bound(self):
        """Test to MIS degree lower bound."""
        g = nx.Graph()
        g.add_node(range(5))
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        g.add_edge(3, 4)

        self.assertAlmostEqual(degree_based_lower_bound(g), 3.0)

    def test_node_edge_based_upper_bound(self):
        """Test to MIS node edge upper bound."""
        g = nx.Graph()
        g.add_node(range(5))
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        g.add_edge(3, 4)

        self.assertAlmostEqual(node_edge_based_upper_bound(g), 5.216990566028302)

    def test_load_graph(self):
        """Test to load graph from file."""

        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "graph.gph"
            gstr = "c line graph.\np edge 5 4\ne 1 2\ne 2 3\ne 3 4\ne 4 5"
            p.write_text(gstr)  # binary write
            # Pass a real path to your loader
            graph = load_graph(str(p))

            g = nx.Graph()
            g.add_node(0)
            g.add_node(1)
            g.add_node(2)
            g.add_node(3)
            g.add_node(4)
            g.add_edge(0, 1, weight=1.0)
            g.add_edge(1, 2, weight=1.0)
            g.add_edge(2, 3, weight=1.0)
            g.add_edge(3, 4, weight=1.0)

            self.assertTrue(graphs_equal(g, graph))

    def test_swap_strat(self):
        """Test swap strategy simplification"""
        swapstrat = SwapStrategy(
            CouplingMap.from_line(5), (((0, 1),), ((1, 2),), ((2, 3),), ((3, 4),))
        )
        g = nx.Graph()
        g.add_node(0)
        g.add_node(1)
        g.add_node(2)
        g.add_node(3)
        g.add_node(4)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        g.add_edge(3, 4)

        g.add_edge(0, 2)
        g.add_edge(0, 3)
        g.add_edge(0, 4)

        edges_after_swap = [
            [(0, 1), (1, 2), (2, 3), (3, 4)],
            [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)],
            [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (3, 4)],
            [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4)],
        ]

        graphs = swap_strategy_simplify(swapstrat, g)
        for i, g in enumerate(graphs):
            self.assertEqual(edges_after_swap[i], list(g.edges))
