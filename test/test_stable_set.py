import itertools
from collections import OrderedDict
from unittest import TestCase

import numpy as np
from qeco.stable_set import (
    stable_set_value,
)
from qeco.utils.graph_utils import random_graph
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize


def compute_best(counts, G):
    """Computes best solution out of all shots.

    Args:
        counts: dict
                key as bitstring, val as count
        G: networkx graph

    Returns:
        best: float
                minimum cost function
        partition: string
                solution bitstring for graph
    """
    best = 0
    partition = []
    for bitstring, _ in counts.items():
        bitstring = [int(i) for i in bitstring]
        obj = stable_set_value(bitstring, G)
        if obj < best:
            best = obj
            partition = [str(s) for s in bitstring]

    return best, partition


def brute_force(G):
    """TODO."""
    # compute all permutations of 0's and 1's given number of nodes in G
    # compute objective function for all permutations and return the best
    all_perms = list(itertools.product([0, 1], repeat=len(G.nodes())))
    max_set = 0
    for bitstring in all_perms:
        obj = stable_set_value(bitstring, G)
        if obj < max_set:
            max_set = obj
    return max_set


def compute_expectation(counts, G):
    """Computes expectation value based on measurement results.

    Args:
        counts: dict
                key as bitstring, val as count
        G: networkx graph

    Returns:
        avg: float
             expectation value
    """
    avg = 0
    sum_count = 0
    for bitstring, count in counts.items():
        bitstring = [int(i) for i in bitstring]
        obj = stable_set_value(bitstring, G)
        avg += obj * count
        sum_count += count

    return avg / sum_count


def get_cost_operator(graph):
    """Generate Hamiltonian for the maximum stable set in a graph.

    Args:
        graph (numpy.ndarray): list of edges making up a graph.

    Returns:
        SparsePauliOp: operator for the Hamiltonian.

    """
    pauli_list = []
    for edge in graph.edges():
        paulis = ["I"] * len(graph)
        paulis[edge[0]], paulis[edge[1]] = "Z", "Z"
        pauli_list.append(("".join(paulis)[::-1], 1.0))
    for i in graph.nodes():
        paulis = ["I"] * len(graph)
        paulis[i] = "Z"
        degree = graph.degree(i)
        pauli_list.append(("".join(paulis)[::-1], degree - 1 / 2))
    # TODO: from_sparse_list?
    return SparsePauliOp.from_list(pauli_list)


def sample_most_likely(state_vector):
    """Compute the most likely binary string from state vector.
    Args:
        state_vector (numpy.ndarray or dict): state vector or counts.
    Returns:
        numpy.ndarray: binary string as numpy.ndarray of ints.
    """
    if isinstance(state_vector, (OrderedDict, dict)):
        # get the binary string with the largest count
        binary_string = sorted(state_vector.items(), key=lambda kv: kv[1])[-1][0]
        x = np.asarray([int(y) for y in reversed(list(binary_string))])
        return x
    else:
        n = int(np.log2(state_vector.shape[0]))
        k = np.argmax(np.abs(state_vector))
        x = np.zeros(n)
        for i in range(n):
            x[i] = k % 2
            k >>= 1
        return 1 - x


class TestStableSet(TestCase):
    def setUp(self):
        self.num_nodes = 5
        self.graph = random_graph(self.num_nodes, edge_prob=0.5)
        self.op = get_cost_operator(self.graph)

    def test_stable_set(self):
        """Test to find the minimum eigenvalue and stable set solution."""
        # find minimum eigenvalue and corresponding eigenvector
        eigenvalues, eigenvectors = np.linalg.eigh(self.op)
        min_eigenvalue = eigenvalues[0]
        ground_state = eigenvectors[:, 0]

        # convert the ground state to a binary solution
        x = sample_most_likely(ground_state)

        self.assertAlmostEqual(min_eigenvalue, -4.5)
        # 1 in bitstring represents inclusion in the stable set
        np.testing.assert_array_equal(x, [1.0, 0.0, 1.0, 0.0, 1.0])
        # value of stable_set_value is objective function minima
        self.assertEqual(stable_set_value(x, self.graph), -3.0)

    def test_qaoa_optimisation(self):
        """Test QAOA optimisation to find stable set solution using COBYLA."""
        # define your QAOA circuit
        p = 2
        circuit = QAOAAnsatz(cost_operator=self.op, reps=p)
        circuit.measure_all()
        backend = StatevectorSampler()

        # result function for the optimiser
        def get_result(G, circuit):
            def execute_circ(params):
                bc = circuit.assign_parameters(params)
                counts = backend.run([bc], shots=1000).result()[0].data.meas.get_counts()
                expect = compute_expectation(counts, G)
                return expect

            return execute_circ

        # initialise guess for parameters
        init_guess = np.random.normal(0, 2.0 * np.pi, 2 * p)
        expectation = get_result(self.graph, circuit)
        # run COBYLA
        res = minimize(expectation, init_guess, method="COBYLA")
        # apply optimised parameters to the circuit
        bc = circuit.assign_parameters(res.x)
        counts = backend.run([bc], shots=1000).result()[0].data.meas.get_counts()

        # find the best and target bitstring
        best = compute_best(counts, self.graph)
        target = brute_force(self.graph)
        self.assertAlmostEqual(best[0], target)
