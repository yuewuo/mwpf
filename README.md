# MWPF
### Hypergraph <span style="color: red; font-size: 120%;">M</span>inimum-<span style="color: red; font-size: 120%;">W</span>eight <span style="color: red; font-size: 120%;">P</span>arity <span style="color: red; font-size: 120%;">F</span>actor Decoder for QEC

[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=for-the-badge)](https://unitary.fund)

*Preview claim: We publish the Python package that may subject to breaking changes. The source code and the full version will be made publicly available with our coming paper.*

Hypergraph MWPF is proven to be **NP-hard** [1]. Our design is taking advantage of clustering technique to lower
the **average** time complexity and reach **almost-linear** average time complexity at small physical error rate.
Please wait for our paper for more discussion of the speed v.s. accuracy.

[<img src="https://raw.githubusercontent.com/yuewuo/conference-talk-2024-APS-march-meeting/main/video_maker/small_color_code_example.gif" width="50%" alt="Color Code Example (click for YouTube video)" align="center" loop=infinite>](https://youtu.be/26jgRb669UE)

## Installation

```sh
pip install -U mwpf
```

MWPF now supports [stim](https://github.com/quantumlib/Stim) simulation. You may experience high logical error rate due to some [adaptor bugs](https://github.com/quantumlib/Stim/pull/873) prior to [sinter](https://pypi.org/project/sinter) version 1.15, so instead of using the `mw_parity_factor` decoder in `sinter`, you may want to manually import the latest MWPF decoder from the `mwpf>=0.2.8` package.

```sh
pip install -U 'mwpf[stim]'
```

```python
from mwpf import SinterMWPFDecoder

sinter.collect(
    tasks = ...,
    num_workers = 1,
    decoders = ["mwpf"],
    custom_decoders = { "mwpf": SinterMWPFDecoder(cluster_node_limit=50) },
)

# MWPF decoder now supports heralded error decoding in stim circuit!
#   Since the sinter.Decoder interface only gives a DEM (Detector Error Model), which doesn't include heralded error
#   information, you need to manually provide the circuit when constructing the decoder, like below
custom_decoders = { "mwpf": SinterMWPFDecoder(cluster_node_limit=50).with_circuit(circuit) }
```


## Background

The Most-Likely Error (MLE) decoding problem can be formulated as a MWPF problem.

![](https://raw.githubusercontent.com/yuewuo/conference-talk-2024-APS-march-meeting/main/images/MLE_decoding.png)

![](https://raw.githubusercontent.com/yuewuo/conference-talk-2024-APS-march-meeting/main/images/MWPF_definition.png)

#### Naming

We named it Minimum-Weight Parity Factor because of a concept called "parity $(g,f)$-factor" in Lovász's 1972 paper [2]. In the context of QEC, the functions $g(v)$ and $f(v)$ associates with the measured syndrome as shown above. Without ambiguity, we just call it "parity factor".

#### Relationship with MWPM decoder

Minimum-Weight Perfect Matching decoder is the traditional decoder for QEC. When the decoding graph is a simple graph, the MLE decoding problem can be reduced to an MWPM problem on a generated complete graph with $O(|V|^2)$ edges. Two recent works [2] and [3] improves the average time complexity to almost theoretical upper bound of $O(|V|)$ by not explicitly generating the complete graph. This motivates our idea to design an algorithm directly on the model graph, hoping to reach the same $O(|V|)$ average time complexity even if the model graph is hypergraph.

#### Key Idea

We try to extend the blossom algorithm that solves the MWPM problem on simple graphs. An interesting attribute of the blossom algorithm is that it deals with an **exponential** number of linear constraints in order to relax the integer constraints. The added linear constraints, which we refer to as "blossom constraints", are based on a very simple idea: filtering out invalid solutions. The blossom constraints simply say "any odd cluster cannot be perfectly matched internally", which is obviously true. However, this "filtering" requires an exponential number of linear constraints [5], which is impossible to generate efficiently. Thus, the algorithm must **implicitly** consider those exponential number of constraints without ever listing them. In fact, the blossom algorithm only keeps track of $O(|V|)$ such constraints from the exponentially many. Surprisingly, this works! Inspired by this miracle, we now have the courage to use an exponential number of linear constraints to solve MWPF.

#### Rigorous Math

![](https://raw.githubusercontent.com/yuewuo/conference-talk-2024-APS-march-meeting/main/images/MWPF_to_ILP.png)

The ILP problem above is very similar to that of the blossom algorithm, except for the more complex "blossom"definition: it's now a subgraph $S=(V_S, E_S), V_S \subseteq V, E_S \subseteq E$ instead of just a subset of vertices $S\subseteq V$ in the blossom algorithm. We have **mathematically** proved the equivalence between hypergraph MWPF and this ILP. In the next step, we simply relax the integer constraints.

![](https://raw.githubusercontent.com/yuewuo/conference-talk-2024-APS-march-meeting/main/images/ILP_to_LP.png)

Clearly, as a relaxation, the minimum objective value is no larger than that of the ILP. Unfortunately, we haven't been able to rigorously prove that they have the same optimal objective value, nor can we find a counter example. We leave this conjecture as is for now, and do not assume its correctness.

##### Conjecture: $\min\text{ILP}=\min\text{LP}$. 

![](https://raw.githubusercontent.com/yuewuo/conference-talk-2024-APS-march-meeting/main/images/LP_to_DLP.png)

The dual problem is a transpose of the primal LP problem. According to the duality theorem, they have the same optimal value. The key is that each primal constraint becomes a dual variable, and each primal variable becomes a dual constraint. Clearly, for the dual problem, $y_S = 0, \forall S \in \mathcal{O}$ is a solution (despite usually not the optimal solution). In this way, we can keep track of only **non-zero** dual variables to implicitly considering all the exponentially many primal constraints. Since the dual LP problem becomes a maximization problem, we have the whole inequality chain as below.

![](https://raw.githubusercontent.com/yuewuo/conference-talk-2024-APS-march-meeting/main/images/inequality_chain.png)

If we can find a pair of feasible primal and dual solutions with the same weight, then this inequality chain **collapses** to an equality chain and the primal solution is **proven to be optimal**. Even if they are not equal, we still get a proximity bound.

In fact, MWPM decoder using the blossom algorithm fits in this framework, where the alternating tree operation guarantees that in the end the primal must be equal to the dual. Union-Find decoder and hypergraph UF decoder can also be explained by this framework, but the proximity bound is usually not singleton.

## Usage

The decoding process is two steps (shown in [Background](#background))

1. offline: construct the model graph given the QEC code and the noise model
2. online: compute the most-likely error $\mathcal{E}\subseteq E$ given the syndrome (represented by the defect vertices $D \subseteq V$) and some dynamic weights

```python
from mwpf import HyperEdge, SolverInitializer, SolverSerialJointSingleHair, SyndromePattern

# Offline
vertex_num = 4
weighted_edges = [
    HyperEdge([0, 1], 100),  # [vertices], weight
    HyperEdge([1, 2], 100),
    HyperEdge([2, 3], 100),
    HyperEdge([0], 100),  # boundary vertex
    HyperEdge([0, 1, 2], 60),  # hyperedge
]
initializer = SolverInitializer(vertex_num, weighted_edges)
hyperion = SolverSerialJointSingleHair(initializer)

# Online
syndrome = [0, 1, 3]
hyperion.solve(SyndromePattern(syndrome))
hyperion_subgraph = hyperion.subgraph()
print(hyperion_subgraph)  # out: [3, 5], weighted 160
_, bound = hyperion.subgraph_range()
print((bound.lower, bound.upper))  # out: (Fraction(160, 1), Fraction(160, 1))
```

The example hyergraph is below: grey edges are weighted 100 and the purple hyperedge is weighted 60.

![](https://raw.githubusercontent.com/yuewuo/conference-talk-2024-APS-march-meeting/main/images/example_hypergraph.png)

In constrast, if we were to solve MWPF with MWPM decoder, then we have to ignore the hyperedge $e_4$ and thus leads to suboptimal result, as shown by the following Python script using the [Fusion Blossom](https://pypi.org/project/fusion-blossom/) library.

```python
from fusion_blossom import SolverInitializer, SolverSerial, SyndromePattern

# Offline
vertex_num = 5
weighted_edges = [(0, 4, 100), (0, 1, 100), (1, 2, 100), (2, 3, 100)]
virtual_vertices = [4]
initializer = SolverInitializer(vertex_num, weighted_edges, virtual_vertices)
fusion = SolverSerial(initializer)

# Online
syndrome = [0, 1, 3]
fusion.solve(SyndromePattern(syndrome))
fusion_subgraph = fusion.subgraph()
print(fusion_subgraph)  # out: [0, 2, 3], which is weighted 300 instead of 160
```

## Advanced Usage

When trading off accuracy and decoding time, we provide a timeout parameter for the decoder. Also, one can specify whether you want the clusters to all grow together or grow one by one. More parameters are coming as we develop the library.

```python
config = {
    "cluster_node_limit":  50,  # how many dual variables are allowed in each cluster before falling back to UF decoder,
                                # by default infinite but setting it to 50 works for circuit-level surface code d=7
                                # for millisecond decoding. I would recommend use this option alone (without timeout) to
                                # tune between decoding speed and accuracy.
    "timeout": 3.0,  # 3 second timeout for each cluster, by default infinite (preferred)
}
hyperion = SolverSerialJointSingleHair(initializer, config)
```

An embedded visualization tool is coming soon.

## Examples

For surface code with depolarizing noise mode $p_x =p_y=p_z = p/3$, here shows physical error rates 1%, 2% and 4% (left to right).

[<img src="https://raw.githubusercontent.com/yuewuo/conference-talk-2024-APS-march-meeting/main/video_maker/surface_code_example.gif" alt="Surface Code Example (click for YouTube video)" align="center" loop=infinite>](https://youtu.be/SjZ8rMdYZ54)

For triangle color code with X errors, here shows physical error rates 1%, 2% and 4% (left to right).

[<img src="https://raw.githubusercontent.com/yuewuo/conference-talk-2024-APS-march-meeting/main/video_maker/triangle_color_code_example.gif" alt="Triangle Color Code Example (click for YouTube video)" align="center" loop=infinite>](https://youtu.be/1KN62fmR7OM)

For circuit-level surface code, here shows physical error rate 0.03%, 0.1% and 0.3% (left to right).

[<img src="https://raw.githubusercontent.com/yuewuo/conference-talk-2024-APS-march-meeting/main/video_maker/circuit_level_example.gif" alt="Circuit-level Surface Code Example (click for YouTube video)" align="center" loop=infinite>](https://youtu.be/ki9fHiA4Gyo)

## Reference

[1] Berlekamp, Elwyn, Robert McEliece, and Henk Van Tilborg. "On the inherent intractability of certain coding problems (corresp.)." IEEE Transactions on Information Theory 24.3 (1978): 384-386.

[2] Lovász, László. "The factorization of graphs. II." Acta Mathematica Academiae Scientiarum Hungarica 23 (1972): 223-246.

[3] Higgott, Oscar, and Craig Gidney. "Sparse Blossom: correcting a million errors per core second with minimum-weight matching." arXiv preprint arXiv:2303.15933 (2023).

[4] Wu, Yue, and Lin Zhong. "Fusion Blossom: Fast MWPM Decoders for QEC." arXiv preprint arXiv:2305.08307 (2023).

[5] Rothvoß, Thomas. "The matching polytope has exponential extension complexity." *Journal of the ACM (JACM)* 64.6 (2017): 1-19.

