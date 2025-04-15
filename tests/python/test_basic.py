from common import *
import fusion_blossom as fb
import numpy as np


def prepare_hyperion_solver() -> mwpf.SolverSerialJointSingleHair:
    vertex_num = 6
    weighted_edges = [
        mwpf.HyperEdge([0, 1], 100),
        mwpf.HyperEdge([1, 2], 100),
        mwpf.HyperEdge([2, 3], 100),
        mwpf.HyperEdge([3, 4], 100),
        mwpf.HyperEdge([4, 5], 100),
        mwpf.HyperEdge([1, 2, 3], 60),  # hyper edge
        mwpf.HyperEdge([0], 0),  # virtual vertex
        mwpf.HyperEdge([5], 0),  # virtual vertex
    ]
    initializer = mwpf.SolverInitializer(vertex_num, weighted_edges)
    solver = mwpf.SolverSerialJointSingleHair(initializer)
    return solver


def prepare_fusion_solver() -> fb.SolverSerial:
    vertex_num = 6
    weighted_edges = [(0, 1, 100), (1, 2, 100), (2, 3, 100), (3, 4, 100), (4, 5, 100)]
    virtual_vertices = [0, 5]
    initializer = fb.SolverInitializer(vertex_num, weighted_edges, virtual_vertices)
    solver = fb.SolverSerial(initializer)
    return solver


def test_compare_hyperion_fusion() -> None:
    syndrome = [1, 2, 4]
    # hyperion
    hyperion = prepare_hyperion_solver()
    hyperion.solve(mwpf.SyndromePattern(syndrome))
    hyperion_subgraph = hyperion.subgraph()
    _, bound = hyperion.subgraph_range()
    print(hyperion_subgraph)
    print((bound.lower, bound.upper))
    # fusion blossom
    fusion = prepare_fusion_solver()
    fusion.solve(fb.SyndromePattern(syndrome))
    fusion_subgraph = fusion.subgraph()
    print(fusion_subgraph)


def test_initializer_uniform_weight():
    code = mwpf.CodeCapacityColorCode(d=5, p=0.01)
    initializer = code.get_initializer()
    initializer.uniform_weights(weight=1)
    for edge in initializer.weighted_edges:
        assert edge.weight.float() == 1


def test_numpy_array_parse():
    # pytest -s tests/python/test_basic.py::test_numpy_array_parse
    syndrome = np.array([1, 2, 4], dtype="int32")
    mwpf.SyndromePattern(syndrome)


def test_non_connected_graphs():
    vertex_num = 6
    weighted_edges = [mwpf.HyperEdge([1, 2, 3], 60)]
    initializer = mwpf.SolverInitializer(vertex_num, weighted_edges)
    solver = mwpf.Solver(initializer)
    print(solver.get_initializer())
    solver.solve(mwpf.SyndromePattern([1, 2, 3]))
    subgraph = solver.subgraph()
    print(subgraph)
    assert subgraph == [0]
    solver.clear()
    solver.solve(mwpf.SyndromePattern([]))
    subgraph = solver.subgraph()
    print(subgraph)
    assert subgraph == []
