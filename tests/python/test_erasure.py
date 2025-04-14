from test_basic import *


def test_erasure_decoding():
    vertex_num = 3
    weighted_edges = [
        mwpf.HyperEdge([0, 1], 100),
        mwpf.HyperEdge([1, 2], 100),
        mwpf.HyperEdge([2, 0], 100),
    ]
    initializer = mwpf.SolverInitializer(vertex_num, weighted_edges)
    solver = mwpf.SolverSerialJointSingleHair(initializer)
    solver.solve(mwpf.SyndromePattern([0, 2]))
    subgraph, bound = solver.subgraph_range()
    print(subgraph)
    print((bound.lower, bound.upper))
    assert subgraph == [2]
    assert bound.lower == bound.upper
    assert bound.upper == 100
    # however if I set the first two edges as erasures, it should choose the first two edges
    # because their weights becomes 0
    solver.solve(mwpf.SyndromePattern([0, 2], erasures=[0, 1]))
    subgraph, bound = solver.subgraph_range()
    print(subgraph)
    print((bound.lower, bound.upper))
    assert subgraph == [0, 1]
    assert bound.lower == bound.upper
    assert bound.upper == 0


def test_override_weights_decoding():
    vertex_num = 3
    weighted_edges = [
        mwpf.HyperEdge([0, 1], 100),
        mwpf.HyperEdge([1, 2], 100),
        mwpf.HyperEdge([2, 0], 100),
    ]
    initializer = mwpf.SolverInitializer(vertex_num, weighted_edges)
    solver = mwpf.SolverSerialJointSingleHair(initializer)
    # force set weights to [0, 0, 100]
    solver.solve(
        mwpf.SyndromePattern(
            [0, 2],
            override_weights=[0, 0, 100],
            override_ratio=1,
        )
    )
    subgraph, bound = solver.subgraph_range()
    print(subgraph)
    print((bound.lower, bound.upper))
    assert subgraph == [0, 1]
    assert bound.lower == bound.upper
    assert bound.upper == 0
    # mix weights of ratio = 0.8
    # 0.8 * [0, 0, 100] + 0.2 * [100, 100, 100] = [20, 20, 100]
    solver.solve(
        mwpf.SyndromePattern(
            [0, 2],
            override_weights=[0, 0, 100],
            override_ratio=0.8,
        )
    )
    subgraph, bound = solver.subgraph_range()
    print(subgraph)
    print((bound.lower, bound.upper))
    assert subgraph == [0, 1]
    assert bound.lower == bound.upper
    assert bound.upper.approx_eq(40)
    # set to negative weight
    solver.solve(
        mwpf.SyndromePattern(
            [0, 2],
            override_weights=[-20, 10, 100],
        )
    )
    subgraph, bound = solver.subgraph_range()
    print(subgraph)
    print((bound.lower, bound.upper))
    assert subgraph == [0, 1]
    assert bound.lower == bound.upper
    assert bound.upper == -10
    # if I do not mix the weights that much, the optimal solution is still [2]
    solver.solve(
        mwpf.SyndromePattern(
            [0, 2],
            override_weights=[-20, 10, 100],
            override_ratio=0.1,
        )
    )
    subgraph, bound = solver.subgraph_range()
    print(subgraph)
    print((bound.lower, bound.upper))
    assert subgraph == [2]
    assert bound.lower == bound.upper
    assert bound.upper == 100
    # mixing the negative weights enough and then the optimal solution changes
    # 0.5 * [-20, 10, 100] + 0.5 * [100, 100, 100] = [40, 55, 100]
    solver.solve(
        mwpf.SyndromePattern(
            [0, 2],
            override_weights=[-20, 10, 100],
            override_ratio=0.5,
        )
    )
    subgraph, bound = solver.subgraph_range()
    print(subgraph)
    print((bound.lower, bound.upper))
    assert subgraph == [0, 1]
    assert bound.lower == bound.upper
    assert bound.upper == 95


def test_heralded_decoding():
    vertex_num = 3
    weighted_edges = [
        mwpf.HyperEdge([0, 1], 1),
        mwpf.HyperEdge([1, 2], 1),
        mwpf.HyperEdge([2, 0], 1),
    ]
    heralds = [
        # add another set of errors with 50% error on edge 0 and 50% error on edge 1
        [(0, 0), (1, 0)],
        # some probability mix
        [(0, 0.2), (1, 0.3)],
        [(0, 0.6), (1, 0.5)],
        {0: -0.2, 1: -0.3},
    ]
    initializer = mwpf.SolverInitializer(vertex_num, weighted_edges, heralds=heralds)
    print(initializer)
    assert initializer.heralds == [
        {0: 0, 1: 0},
        {0: 0.2, 1: 0.3},
        {0: 0.6, 1: 0.5},
        {0: -0.2, 1: -0.3},
    ]
    # when the first herald error happens, it adds two 0-weighted edges
    solver = mwpf.SolverSerialJointSingleHair(initializer)
    solver.solve(
        mwpf.SyndromePattern(
            [0, 2],
            heralds=[0],
        )
    )
    subgraph, bound = solver.subgraph_range()
    print(subgraph)
    print((bound.lower, bound.upper))
    assert subgraph == [0, 1]
    assert bound.lower == bound.upper
    assert bound.upper == 0
    # when the second herald error happens, it adds weight 0.2 and weight 0.3 edges
    # mixing weights 0.2 and 1 gives us log((1 + exp(0.2) * exp(1)) / (exp(0.2) + exp(1))) = 0.092181801390
    #     one can verify that by p1 = 1 / (1 + exp(0.2)) = 45%, p2 = 1 / (1 + exp(1)) = 27%
    #        p' = p1 (1-p2) + p2 (1-p1) = 47.7%, w' = log((1-p')/p') ~= 0.092
    # mixing weights 0.3 and 1 gives us log((1 + exp(0.3) * exp(1)) / (exp(0.3) + exp(1))) = 0.13782240494
    # two edges together weight 0.23000420633778762
    solver = mwpf.SolverSerialJointSingleHair(initializer)
    solver.solve(
        mwpf.SyndromePattern(
            [0, 2],
            heralds=[1],
        )
    )
    subgraph, bound = solver.subgraph_range()
    print(subgraph)
    print((bound.lower, bound.upper))
    assert subgraph == [0, 1]
    assert bound.lower == bound.upper
    assert bound.upper == 0.23000420633778762
    # try some negative edges
    solver = mwpf.SolverSerialJointSingleHair(initializer)
    solver.solve(
        mwpf.SyndromePattern(
            [0, 2],
            heralds=[3],
        )
    )
    subgraph, bound = solver.subgraph_range()
    print(subgraph)
    print((bound.lower, bound.upper))
    assert subgraph == [0, 1]
    assert bound.lower == bound.upper
    assert bound.upper == -0.23000420633778762


def test_exclusive_weight_sum():
    assert mwpf.exclusive_weight_sum(0.2, 1) == 0.09218180139025334
    assert mwpf.exclusive_weight_sum(0.3, 1) == 0.13782240494753428
    assert mwpf.exclusive_weight_sum(0.3, 0) == 0
    assert mwpf.exclusive_weight_sum(0, 0.3) == 0
    assert mwpf.exclusive_weight_sum(-1, 0.3) == -0.13782240494753428
    assert mwpf.exclusive_weight_sum(-1, 0) == 0
    assert mwpf.exclusive_weight_sum(-0.2, 1) == -0.09218180139025334
    assert mwpf.exclusive_weight_sum(-0.3, 1) == -0.13782240494753428
