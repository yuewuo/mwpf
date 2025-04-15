from common import *
import pickle


def test_basic_pickle():
    vertex_num = 2
    weighted_edges = [
        mwpf.HyperEdge([0, 1], 100),
    ]
    initializer = mwpf.SolverInitializer(vertex_num, weighted_edges)
    solver = mwpf.SolverSerialJointSingleHair(initializer)

    pickled = pickle.dumps(solver)
    config = solver.config
    print(config)
    print(solver)
    print(pickled)

    print("\n\n#### pickle and then unpickle  ####\n\n")

    solver2 = pickle.loads(pickled)
    config2 = solver2.config
    print(config2)
    print(solver2)

    assert config == config2
