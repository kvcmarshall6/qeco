# (C) Copyright IBM 2026.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Functions for working with stable set problems."""

import networkx as nx
import numba as nb
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import PauliEvolutionGate, QAOAAnsatz
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.primitives import StatevectorSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import Sampler
from qopt_best_practices.transpilation import qaoa_swap_strategy_pm
from qopt_best_practices.transpilation.qaoa_construction_pass import (
    QAOAConstructionPass,
)


# NOTE: this function has been borrowed from stable_set_benchmarking repo
def compute_violations(sample: list[int], graph: nx.Graph) -> dict:
    """Compute the number of violations of the given node.

    If the node is in the stable set then for each of its neighbors
    compute the number of neighbors that are also in the stable set.
    """
    violations = dict()
    for node in graph.nodes():
        num_violations = 0
        for neighbor in graph.neighbors(node):
            # if node == 0 we add 0 if it is 1 we add 1 if the neighbor is 1.
            if sample[neighbor] == 1:
                num_violations += sample[node]

        violations[node] = num_violations

    return violations


def soften_bitstring(bitstring, epsilon=0.01):
    """Map 0 and 1 to softened values in range (0,1)."""
    return [epsilon if b == "0" else 1 - epsilon for b in bitstring]


def construct_init_state(circuit, input_bitstring):
    """Create warm-start initial state based on softened bitstring approximation."""
    num_qubits = circuit["circuit_to_sample"].num_qubits
    assert (
        len(input_bitstring) == num_qubits
    ), "Bitstring length must match number of qubits."

    init_state = QuantumCircuit(num_qubits)

    # compute rotation angles from relaxed values
    thetas = [2 * np.arcsin(np.sqrt(c)) for c in input_bitstring]

    for i, theta in enumerate(reversed(thetas)):
        init_state.ry(theta, i)

    return init_state


def construct_init_state_params(circuit):
    """Create warm-start initial state based on softened bitstring approximation."""
    num_qubits = circuit["circuit_to_sample"].num_qubits

    init_state = QuantumCircuit(num_qubits)

    # compute rotation angles from relaxed values
    thetas = ParameterVector("th_init", num_qubits)
    for i, theta in enumerate(reversed(thetas)):
        init_state.ry(theta, i)

    return init_state


def construct_mixer(circuit, input_bitstring):
    """Construct warm-start mixer layer based on softened input bitstring."""
    num_qubits = circuit["circuit_to_sample"].num_qubits
    assert (
        len(input_bitstring) == num_qubits
    ), "Bitstring length must match number of qubits."

    mixer_layer = QuantumCircuit(num_qubits)
    beta = Parameter("β")

    # compute rotation angles from relaxed values
    thetas = [2 * np.arcsin(np.sqrt(c)) for c in input_bitstring]

    for i, theta in enumerate(reversed(thetas)):
        # use (+2 * beta) for unbiased case (epsilon=0.5, theta=np.pi/2) to match rx(2 * beta) in normal QAOA mixer
        sign = 1 if np.isclose(theta, np.pi / 2, atol=1e-4) else -1
        mixer_layer.ry(-theta, i)
        mixer_layer.rz(sign * 2 * beta, i)
        mixer_layer.ry(theta, i)

    return mixer_layer


def construct_mixer_params(circuit):
    """Construct warm-start mixer layer based on softened input bitstring."""
    num_qubits = circuit["circuit_to_sample"].num_qubits

    mixer_layer = QuantumCircuit(num_qubits)

    beta = Parameter("β")
    # start next param name with \beta -- ugly hack to set beta as first param of mixer_layer, assumed elsewhere; in .run of QAOAConsutrctionPass
    thetas = ParameterVector("βth_mixer", num_qubits)
    signs = ParameterVector("βsigns", num_qubits)

    for i, theta in enumerate(reversed(thetas)):
        mixer_layer.ry(-theta, i)
        mixer_layer.rz(signs[i] * 2 * beta, i)
        mixer_layer.ry(theta, i)

    return mixer_layer


def construct_warm_start_circuit(
    init_state, mixer_layer, old_circuits_dict, params, backend, layers=2
):
    """Warm start QAOA circuit."""
    new_circuits_dict = dict()
    my_properties = {}

    construction_pass = QAOAConstructionPass(layers, init_state, mixer_layer)
    construction_pass.property_set = (
        my_properties  # merge back-in the permutation.
    )
    transpiled_circ = dag_to_circuit(
        construction_pass.run(circuit_to_dag(old_circuits_dict["cost_circuit"]))
    )

    circ_to_sample = transpiled_circ.assign_parameters(params, inplace=False)

    new_circuits_dict["cost_circuit"] = old_circuits_dict["cost_circuit"]
    new_circuits_dict["circuit_to_sample"] = circ_to_sample

    if backend is not None:
        pm1 = generate_preset_pass_manager(
            optimization_level=3,
            backend=backend,
            scheduling_method="alap",
        )
        new_circuits_dict["backend"] = pm1.run(circ_to_sample)

        metadata = {}
        metadata["parameters"] = params
        new_circuits_dict["backend"].metadata = metadata

    return new_circuits_dict


# NOTE: this function has been borrowed from stable_set_benchmarking repo
def circuit_construction(
    singles,
    doubles,
    params,
    backend,
    swap_strat,
    edge_coloring,
    layers,
    metadata=None,
):
    """Construct circuits."""
    circuits_dict = dict()

    n = len(doubles[0].paulis[0])

    # Setup a cost op with the two-qubit terms only.
    doubles_circ = QAOAAnsatz(
        doubles,
        initial_state=QuantumCircuit(n),
        mixer_operator=QuantumCircuit(n),
    )

    circuits_dict["doubles"] = doubles_circ

    # Apply the swap strategy
    my_properties = {}  # We will need this later on.

    def get_permutation(pass_, dag, time, property_set, count):
        my_properties["virtual_permutation_layout"] = property_set[
            "virtual_permutation_layout"
        ]

    config = {
        "num_layers": layers,
        "swap_strategy": swap_strat,
        "edge_coloring": edge_coloring,
        "construct_qaoa": False,
    }

    pm = qaoa_swap_strategy_pm(config)
    tdoubles_circ = pm.run(doubles_circ, callback=get_permutation)
    circuits_dict["tdoubles"] = tdoubles_circ

    singles_circ = QuantumCircuit(n)
    singles_circ.append(
        PauliEvolutionGate(singles, time=2 * tdoubles_circ.parameters[0]),
        range(n),
    )
    tsingles = transpile(singles_circ, basis_gates=["rz"])
    cost_circ = tsingles.compose(tdoubles_circ, inplace=False)
    circuits_dict["cost_circuit"] = cost_circ

    # Finally, construct the full QAOA circuit.
    construction_pass = QAOAConstructionPass(layers)
    construction_pass.property_set = (
        my_properties  # merge back-in the permutation.
    )
    transpiled_circ = dag_to_circuit(
        construction_pass.run(circuit_to_dag(cost_circ))
    )

    circ_to_sample = transpiled_circ.assign_parameters(params, inplace=False)

    circuits_dict["circuit_to_sample"] = circ_to_sample

    if backend is not None:
        pm1 = generate_preset_pass_manager(
            optimization_level=3,
            backend=backend,
            scheduling_method="alap",
        )
        circuits_dict["backend"] = pm1.run(circ_to_sample)

        metadata = {}
        metadata["parameters"] = params
        circuits_dict["backend"].metadata = metadata

    return circuits_dict


def compute_non_maximality(x, g):
    """Count how many nodes could be added to x without violating independence."""
    non_maximal_count = 0
    for node in g.nodes():
        if x[node] == 0:
            neighbors = g.neighbors(node)
            if all(x[neighbor] == 0 for neighbor in neighbors):
                non_maximal_count += 1
    return non_maximal_count


def stable_set_value(x, g, penalty=1.1, maximality_penalty=2.0):
    """Compute the value of a stable set (lower is better).

    Args:
        x (np.ndarray): binary string representing graph partition
        g (rx.PyGraph): graph.
        penalty (float): penalty for edge violations (default set to 1.1).
        maximality_penalty (float): penalty for being non-maximal (default set to 2.0).
    """
    obj = -np.sum(x)

    # compute the number of violations for each node
    violations = compute_violations(x, g)
    total_violations = sum(violations.values()) / 2  # each edge counted twice
    # add the penalty for each violation
    obj += penalty * total_violations

    # penalize non-maximality (nodes that could be added)
    non_maximal = compute_non_maximality(x, g)
    obj += maximality_penalty * non_maximal

    return obj


def stable_set_value_parallel(
    x_matrix,
    indptr,
    indices,
    edge_u,
    edge_v,
    penalty=1.1,
    maximality_penalty=2.0,
):
    """Compute stable set energies for many bitstrings in parallel."""
    n_samples, n_nodes = x_matrix.shape
    energies = np.empty(n_samples, dtype=np.float64)

    for s in nb.prange(n_samples):
        x = x_matrix[s]
        obj = 0.0

        # -sum(x)
        for i in range(n_nodes):
            obj -= x[i]

        # violated edges
        violations = 0
        for e in range(len(edge_u)):
            u = edge_u[e]
            v = edge_v[e]
            if x[u] == 1 and x[v] == 1:
                violations += 1
        obj += penalty * violations

        # non-maximality
        non_maximal = 0
        for i in range(n_nodes):
            if x[i] == 0:
                can_add = 1
                for k in range(indptr[i], indptr[i + 1]):
                    j = indices[k]
                    if x[j] == 1:
                        can_add = 0
                        break
                non_maximal += can_add

        obj += maximality_penalty * non_maximal
        energies[s] = obj

    return energies


def run_experiment(circuit, shots=1, backend=None, session=None):
    """Run experiment."""
    if backend is None:
        backend = StatevectorSampler()
        result = backend.run(
            [circuit["circuit_to_sample"]], shots=shots
        ).result()
        counts = result[0].data.c.get_counts()
    else:
        sampler = Sampler(mode=session)
        job = sampler.run([circuit["backend"]], shots=shots)
        print(f"DEBUG job: {job.job_id()}")
        result = job.result()
        counts = result[0].data.c.get_counts()
    return counts
