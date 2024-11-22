import mwpf
import math, os


def circle_positions(n: int):
    positions = []
    for i in range(n):
        positions.append(
            mwpf.VisualizePosition(
                0.5 + 0.5 * math.cos(2 * math.pi * i / n),
                0.5 + 0.5 * math.sin(2 * math.pi * i / n),
                0,
            )
        )
    return positions


def test_fp_1():
    vertex_num = 4
    weighted_edges = [
        mwpf.HyperEdge([0], 0.6),
        mwpf.HyperEdge([0, 1], 0.7),
        mwpf.HyperEdge([1, 2], 0.8),
        mwpf.HyperEdge([2, 3], 0.9),
        mwpf.HyperEdge([3], 1.1),
        mwpf.HyperEdge([0, 1, 2], 0.3),  # hyper edge
    ]
    initializer = mwpf.SolverInitializer(vertex_num, weighted_edges)
    solver = mwpf.SolverSerialJointSingleHair(initializer)
    visualizer = mwpf.Visualizer(positions=circle_positions(vertex_num))
    syndrome = mwpf.SyndromePattern([0, 1, 2])
    solver.solve(syndrome, visualizer)
    subgraph, bound = solver.subgraph_range(visualizer)
    print(subgraph, bound)
    assert bound.lower == bound.upper
    assert bound.lower.float() == 0.3
    with open(os.path.join(os.path.dirname(__file__), f"test_fp_1.html"), "w") as f:
        f.write(visualizer.generate_html())


def test_fp_2():
    vertex_num = 4
    weighted_edges = [
        mwpf.HyperEdge([0], -0.6),
        mwpf.HyperEdge([0, 1], 0.7),
        mwpf.HyperEdge([1, 2], 0.8),
        mwpf.HyperEdge([2, 3], 0.9),
        mwpf.HyperEdge([3], 1.1),
        mwpf.HyperEdge([0, 1, 2], 0.3),  # hyper edge
    ]
    initializer = mwpf.SolverInitializer(vertex_num, weighted_edges)
    solver = mwpf.SolverSerialJointSingleHair(initializer)
    visualizer = mwpf.Visualizer(positions=circle_positions(vertex_num))
    syndrome = mwpf.SyndromePattern([0, 1, 2])
    solver.solve(syndrome, visualizer)
    subgraph, bound = solver.subgraph_range(visualizer)
    print(subgraph, bound)
    with open(os.path.join(os.path.dirname(__file__), f"test_fp_1.html"), "w") as f:
        f.write(visualizer.generate_html())
