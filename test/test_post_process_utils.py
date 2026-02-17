from collections import defaultdict
from unittest import TestCase

import networkx as nx
from qeco.stable_set import circuit_construction, compute_violations, run_experiment
from qeco.utils.graph_utils import load_graph, random_graph
from qeco.utils.post_process_utils import boost_samples, run_metropolis_hastings
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import (
    SwapStrategy,
)
from qiskit_optimization.applications import StableSet
from qiskit_optimization.converters import QuadraticProgramToQubo


class Test_PostProcessingUtils(TestCase):
    def setUp(self):
        self.num_nodes = 5
        self.graph = random_graph(self.num_nodes, edge_prob=0.5)

    def test_M_H_algorithm(self):
        # first test case should accept and return s1
        bitstring_in = {"00000": 1}  # cost = 0
        bitstring_out = {"10000": 1}  # cost = -1
        returned_bitsting = run_metropolis_hastings(
            bitstring_in,
            bitstring_out,
            self.graph,
            T=1.0,
        )
        self.assertEqual(dict(returned_bitsting), {"10000": 1})

        # second test case should reject and return s0
        bitstring_in = {"10000": 1}
        bitstring_out = {"00000": 1}
        returned_bitsting = run_metropolis_hastings(
            bitstring_in,
            bitstring_out,
            self.graph,
            T=0,
        )
        self.assertEqual(dict(returned_bitsting), {"10000": 1})

        # third test - sometimes reject sometimes accept
        bitstring_in = {"10000": 1}
        bitstring_out = {"00000": 1}
        acc_rej = []
        for _ in range(1000):
            returned_bitsting = run_metropolis_hastings(
                bitstring_in, bitstring_out, self.graph, T=10
            )
            acc_rej.append(bool(dict(returned_bitsting) == {"00000": 1}))
        acc_prob_t1 = sum(acc_rej) / 1000
        self.assertGreater(acc_prob_t1, 0)
        self.assertLess(acc_prob_t1, 1)

        bitstring_in = {"10000": 1}
        bitstring_out = {"00000": 1}
        acc_rej = []
        for _ in range(1000):
            returned_bitsting = run_metropolis_hastings(
                bitstring_in, bitstring_out, self.graph, T=50
            )
            acc_rej.append(bool(dict(returned_bitsting) == {"00000": 1}))
        acc_prob_t5 = sum(acc_rej) / 1000
        self.assertGreater(acc_prob_t5, acc_prob_t1)
        self.assertGreater(acc_prob_t5, 0)
        self.assertLess(acc_prob_t5, 1)

    # @patch("qeco.stable_set.circuit_construction")
    def test_greedy_post_processing(self):
        """Test to check greedy post processing improves stable set solution."""

        graph = load_graph("./data/mammalia-kangaroo-interactions.gph")
        problem = StableSet(graph)
        quadratic_program = problem.to_quadratic_program()

        penalty = 0.10

        # empty lists for single-qubit and two-qubit interactions terms for Ising ham
        singles = []
        doubles = []

        # derive QUBO representations for stable set problem
        qubo = QuadraticProgramToQubo(penalty=penalty).convert(quadratic_program)
        # store Ising hamiltonian derived from QUBOs aka. cost operator and constant energy offset
        op, _ = qubo.to_ising()
        # calculate the total number of Z operators in each term of Ising ham
        # select terms == 1 Z operator (single-qubit / linear interactions)
        singles = op[op.paulis.z.sum(axis=-1) == 1]
        # select terms == 2 Z operator (two-qubit / quadratic interactions)
        doubles = op[op.paulis.z.sum(axis=-1) == 2]
        params = [0.771962866806646, 2.5136301424924747]
        swap_strat = SwapStrategy.from_line(range(graph.order()))
        edge_coloring = {(idx, idx + 1): (idx + 1) % 2 for idx in range(graph.order())}
        circuit = circuit_construction(
            singles, doubles, params, None, swap_strat, edge_coloring, layers=1
        )

        for _ in range(10):
            samples = run_experiment(circuit)
            unboosted_cv, _ = counts_to_cost_val(samples, graph)
            _, boosted_cv = boost_samples(samples, graph)
            # assert boosted cost vals are equal to or greater than the unboosted cvs
            self.assertGreaterEqual(sum(boosted_cv.keys()), sum(unboosted_cv.keys()))


def counts_to_cost_val(
    counts: dict,
    graph: nx.Graph,
) -> tuple[dict[int, float], dict[int, float]]:
    """Convert a dict of counts to a dict of objective values."""
    cost_vals: dict[int, float] = defaultdict(float)
    infeasible: dict[int, float] = defaultdict(float)
    for bit_str, count in counts.items():
        candidate = [int(x) for x in bit_str[::-1]]
        if is_feasible(candidate, graph):
            cost_vals[sum(candidate)] += count
        else:
            infeasible[sum(candidate)] += count

    return cost_vals, infeasible


def is_feasible(sample: list[int], graph: nx.Graph):
    """Determine stable set feasibility.

    We implement this function because it is orders of magnitude faster
    then using `QuadraticProgram.is_feasible`.
    """
    return sum(compute_violations(sample, graph).values()) == 0
