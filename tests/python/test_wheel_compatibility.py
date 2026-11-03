import pickle
import sys
import sysconfig
from concurrent.futures import ThreadPoolExecutor

from common import mwpf, pytest


def decode(_):
    initializer = mwpf.SolverInitializer(2, [mwpf.HyperEdge([0, 1], 1)])
    initializer = pickle.loads(pickle.dumps(initializer))
    solver = mwpf.Solver(initializer)
    solver.solve(mwpf.SyndromePattern([0, 1]))
    assert solver.subgraph() == [0]
    _, bounds = solver.subgraph_range()
    assert bounds.lower.float() == 1
    assert bounds.upper.float() == 1
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
