# (C) Copyright IBM 2026.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Graph utilities."""

import copy
import math

import networkx as nx
import numpy as np


def random_graph(
    n,
    weight_range=10,
    edge_prob=0.3,
    negative_weight=True,
    seed=8123179,
):
    """Generate random graph using networkxx."""
    assert weight_range > 0
    random_gen = np.random.default_rng(seed)

    graph = nx.Graph()
    graph.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            if random_gen.random() <= edge_prob:
                weight = random_gen.integers(1, weight_range)
                if random_gen.random() >= 0.5 and negative_weight:
                    weight *= -1
                graph.add_edge(i, j, weight=weight)

    return graph


def load_graph(file_name):
    """Given a file name, load the graph it contains."""
    with open(file_name) as file:
        lines = [line.rstrip() for line in file]

    edge_nodes = set()
    for line in lines:
        if len(line) == 0:
            continue

        if line[0] == "p":
            header = line.split(" ")
            nnodes = int(header[2])
        if line[0] == "e":
            edge = line.split(" ")
            edge_nodes.add(int(edge[1]))
            edge_nodes.add(int(edge[2]))

    if max(edge_nodes) == nnodes:
        start_idx = 1
    elif max(edge_nodes) + 1 == nnodes:
        start_idx = 0

    adj_mat = np.zeros((nnodes, nnodes))
    for line in lines[1:]:
        if len(line) > 0 and line[0] == "e":
            edge = [int(e) for e in line.split(" ")[1:]]
            adj_mat[edge[0] - start_idx, edge[1] - start_idx] = 1

    graph = nx.from_numpy_array(adj_mat)

    return graph


def degree_based_lower_bound(G):
    """Evaluate approximate lower bound of MIS using degree of the graph G."""
    return sum((1 / (G.degree(v) + 1)) for v in G.nodes)


def node_edge_based_upper_bound(G):
    """Evaluate approximate upper bound of MIS using nodes and edges of the graph G."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    bound = (1 / 2) * (1 + math.sqrt(1 - 8 * m - 4 * n + 4 * n**2))
    return bound


def swap_strategy_simplify(swap_strat, graph: nx.Graph) -> list[nx.Graph]:
    """Generates one new graph per layer of the swap strategy.

    This allows us to create problems that require an increasing number of
    two-qubit gates to run.
    """
    sub_graph, all_sub_graphs = nx.Graph(), []

    for node in graph.nodes():
        sub_graph.add_node(node)

    for idx in range(len(swap_strat)):
        for new_edge in swap_strat.new_connections(idx):
            if graph.get_edge_data(*new_edge) is not None:
                sub_graph.add_edge(*new_edge)

        all_sub_graphs.append(copy.deepcopy(sub_graph))

    return all_sub_graphs
