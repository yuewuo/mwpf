import pickle
import sys
import sysconfig
from concurrent.futures import ThreadPoolExecutor

from common import mwpf, pytest


def decode(_):
    edges = [mwpf.HyperEdge([vertex, vertex + 1], 100) for vertex in range(5)]
    edges += [
        mwpf.HyperEdge([1, 2, 3], 60),
        mwpf.HyperEdge([0], 0),
        mwpf.HyperEdge([5], 0),
    ]
    initializer = mwpf.SolverInitializer(6, edges)
    initializer = pickle.loads(pickle.dumps(initializer))
    solver = mwpf.Solver(initializer)
    solver.solve(mwpf.SyndromePattern([1, 2, 4]))
    assert solver.subgraph() == [3, 5]
    _, bounds = solver.subgraph_range()
    assert bounds.lower.float() == 160
    assert bounds.upper.float() == 160
    solver.clear()
    solver.solve(mwpf.SyndromePattern([]))
    assert solver.subgraph() == []
    return True


def test_wheel_decodes():
    assert decode(0)


def test_independent_solvers_in_threads():
    with ThreadPoolExecutor(max_workers=8) as executor:
        assert all(executor.map(decode, range(32)))


@pytest.mark.skipif(
    not sysconfig.get_config_var("Py_GIL_DISABLED"),
    reason="requires a free-threaded interpreter",
)
def test_import_does_not_enable_gil():
    assert not sys._is_gil_enabled()
