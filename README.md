## Quantum-enhanced Markov chain Monte Carlo for combinatorial optimization

This repo accompanies the publication `Quantum-enhanced Markov chain Monte Carlo (QeMCMC) for combinatorial optimization`. We include all necessary code to replicate the `117` node Maximum Independent Set (MIS) results, which finds the MIS for the graph consisting of a fully connected component of `110` nodes and `7` isolated nodes. We also include the `110` node variant of the same problem, for those interested in the filtered graph with isolated nodes removed. 

### Description

The accompanying paper introduces our algorithm, which is a heuristic implementation of QeMCMC, combined with warm-starts for quantum optimization and parallel tempering, and motivates its application to combinatorial optimization problems. We study MIS problems in this publication, but the workflow is applicable to combinatorial optimization more generally. The paper also covers:

1. Background material for subroutines used in the algorithm, including QeMCMC, warm-starting QAOA and parallel tempering. 
2. Implementation of the algorithm to a `117` node MIS problem (notebooks attached).
3. Comparison studies to classical MCMC implementation of the algorithm.
4. Scaling analysis for increasing problem size.

For the interested reader, we also include supporting material for QeMCMC sampling from the Boltzmann distribution of a 5-node graph toy problem. 

### Structure

- `data`: Example graph data for MIS. 
- `src/qeco`
  - `utils`: Utility functions supporting code execution, including graph simplification utilities and Metropolis-Hastings logic for post-processing samples.
  - `stable_set`: Functionality for solving stable set problems, including circuit building functionality. 
- `test`: Test material to support code functionality. 
- `117_node_graph_opt_parallel_tempering.ipynb`: Notebook that illustrate the optimization algorithm execution.
- `5_node_graph_boltzmann_sampling.ipynb`: Notebook that illustrates QeMCMC for sampling from a Boltzmann distribution of a 5-node graph toy problem.

### Set up

As a prerequisite, please ensure you have a community edition (free tier) CPLEX license for working with the remaining graphs in the `data` folder, found [here](https://www.ibm.com/products/ilog-cplex-optimization-studio/pricing).

```bash
pip install -e .
```

If you plan to do development, be sure to also install the development dependencies:
```bash
pip install -e ".[dev]"
```

Then work through the notebook `117_node_graph_opt_parallel_tempering.ipynb`.

### Contributors and acknowledgement

Kate V. Marshall <kate.marshall@ibm.com>
<br/>
Daniel J. Egger
<br/>
Michael Garn
<br/>
Francesca Schiavello
<br/>
Sebastian Brandhofer
<br/>
Christa Zoufal <ouf@zurich.ibm.com>
<br/>
Stefan Woerner <wor@zurich.ibm.com>
