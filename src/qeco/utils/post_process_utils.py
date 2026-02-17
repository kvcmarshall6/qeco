# (C) Copyright IBM 2026.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Post-processing utilities."""

from collections import defaultdict
from typing import Literal

import networkx as nx
import numpy as np

from qeco.stable_set import compute_violations, stable_set_value


# NOTE: this function has been borrowed from stable_set_benchmarking repo
def greedy_post_process(
    counts: dict[str, float],
    graph: nx.Graph,
    boosting: Literal["complete", "partial"] = "complete",
) -> tuple[dict[str, float], dict[int, float]]:
    """Post process the counts for the stable set problem.

    Args:
        counts: A counts dictionary.
        qp: The quadratic program.
        graph: The graph of the stable set problem.
        boosting: Key to apply complete or partial boosting. If complete is chosen,
            violating nodes will be removed from the solution graph and the suggested
            solution will also be improved upon where possible. If partial is chosen,
            only violating nodes will be removed and no solution optimization will be
            applied.
    """
    post_processed_counts: dict[str, float] = defaultdict(float)
    post_processed_cost_values: dict[int, float] = defaultdict(float)

    for sample_bits, count in counts.items():
        sample = [int(bit) for bit in sample_bits][::-1]

        violations = compute_violations(sample, graph)

        candidates = set(graph.nodes())
        while sum(violations.values()) > 0:
            max_offender = max(*list(violations.items()), key=lambda x: x[1])[0]
            sample[max_offender] = 0
            candidates.remove(max_offender)
            violations = compute_violations(sample, graph)

        new_sample = [val for val in sample]
        if boosting == "complete":
            while len(candidates) > 0:
                node = next(iter(candidates))
                if new_sample[node] == 0:
                    test_sample = [val for val in new_sample]
                    test_sample[node] = 1

                    num_violations = sum(compute_violations(test_sample, graph).values())

                    if num_violations == 0:
                        new_sample[node] = 1

                candidates.remove(node)

        # The sample is guaranteed feasible and the objective of stable set is
        # simply the number of 1s.
        post_processed_cost_values[sum(new_sample)] += count

        # convert the processed sample back to a bitstring
        final_sample = "".join(map(str, new_sample[::-1]))
        post_processed_counts[final_sample] += count

    return post_processed_counts, post_processed_cost_values


def boost_samples(
    samples: dict[str, float],
    graph: nx.Graph,
    boosting: Literal["complete", "partial"] = "complete",
) -> tuple[dict[str, float], dict[int, float]]:
    """TODO.

    Args:
        samples: a counts dictionary.
        graph: The graph of the stable set problem.
        boosting: Key to apply complete or partial boosting. If complete is chosen,
            violating nodes will be removed from the solution graph and the suggested
            solution will also be improved upon where possible. If partial is chosen,
            only violating nodes will be removed and no solution optimization will be
            applied.

    Returns:
        A tuple of dictionaries: the boosted counts and computed objective values.
    """
    boosted_samples, cost_values = greedy_post_process(samples, graph, boosting)
    return boosted_samples, cost_values


# NOTE: this function has been borrowed from stable_set_benchmarking repo
def to_cdf(dist_cut: dict):
    """Convert the distribution to a CDF."""
    shots = sum(dist_cut.values())
    values = sorted(dist_cut.keys())
    cdf = None
    for val in values:
        if cdf is None:
            cdf = [dist_cut[val] / shots]
        else:
            cdf.append(dist_cut[val] / shots + cdf[-1])

    return values, cdf


def compute_best_solution(proposed_solutions, graph, penalty):
    """Evaluate lowest energy solution from all proposals."""
    best_solution = None
    best_E_out = float("inf")
    for proposal, _ in proposed_solutions.items():
        proposed_s_array = np.array([int(bit) for bit in proposal])
        E_out = stable_set_value(proposed_s_array[::-1], graph, penalty)

        if E_out < best_E_out:
            best_E_out = E_out
            best_solution = proposal

    print(
        "Best proposed solution: ",
        "".join(map(str, best_solution)),
        "E_out: ",
        best_E_out,
    )
    return {best_solution: int(proposed_solutions[best_solution])}, best_E_out


def run_metropolis_hastings(
    accepted_solution,
    proposal_distribution,
    graph,
    T: float = 1.0,
    bitstring_freqs=None,
):
    """Accept or reject proposed solutions based on MH acceptance criteria ."""
    if isinstance(accepted_solution, dict):
        accepted_solution_str = next(iter(accepted_solution.keys()))
    else:
        accepted_solution_str = accepted_solution

    if bitstring_freqs is not None:
        for bs, c in proposal_distribution.items():
            bitstring_freqs[bs] += c

    accepted_solution_arr = np.array([int(b) for b in accepted_solution_str[::-1]])
    accepted_solution_E = stable_set_value(accepted_solution_arr, graph)

    # compute proposal energies and acceptance probabilities
    proposals = list(proposal_distribution.keys())

    proposal_E_list: list[float] = []
    for bs in proposals:
        proposal_arr = np.array([int(b) for b in bs[::-1]])
        proposal_E_list.append(stable_set_value(proposal_arr, graph))

    proposal_E = np.asarray(proposal_E_list, dtype=float)

    best_idx = int(np.argmin(proposal_E))
    proposal_str = proposals[best_idx]
    best_proposal_E = float(proposal_E[best_idx])
    delta_E = best_proposal_E - accepted_solution_E

    # acceptance criteria

    if T <= 0:
        accept = delta_E <= 0  # greedy at T=0
    else:
        accept = True if delta_E <= 0 else np.random.rand() < np.exp(-delta_E / T)

    # if T > 0:
    #     A = np.minimum(1.0, np.exp(-delta_E / T))
    # else:
    #     A = (delta_E <= 0).astype(float)

    new_accepted_solution = proposal_str if accept else accepted_solution_str
    new_accepted_E = best_proposal_E if accept else accepted_solution_E
    print(
        "Accepted solution: ",
        new_accepted_solution,
        "E_out: ",
        new_accepted_E,
    )
    return {new_accepted_solution: 1}


def run_parallel_tempering_swap_accept(
    solution_i: dict[str, float],
    solution_j: dict[str, float],
    temp_i: float,
    temp_j: float,
    graph,
    penalty: float = 1.1,
    threshold: int = 700,
) -> bool:
    """Perform Metropolis swap acceptance between neighbouring replicas i and j."""
    str_i, _ = next(iter(solution_i.items()))
    str_j, _ = next(iter(solution_j.items()))
    solution_i_array = np.array([int(b) for b in str_i])[::-1]
    solution_j_array = np.array([int(b) for b in str_j])[::-1]

    E_solution_i = stable_set_value(solution_i_array, graph, penalty)
    E_solution_j = stable_set_value(solution_j_array, graph, penalty)

    beta_i = 1.0 / temp_i
    beta_j = 1.0 / temp_j

    delta = (beta_j - beta_i) * (E_solution_i - E_solution_j)

    if delta <= 0:
        A = 1.0
    elif delta > threshold:
        A = 0.0
    else:
        A = min(1.0, np.exp(-delta))

    return np.random.rand() < A
