from common import *
import traceback


def test_basic_panic():
    vertex_num = 2
    weighted_edges = [
        mwpf.HyperEdge([0, 1], 100),
    ]
    initializer = mwpf.SolverInitializer(vertex_num, weighted_edges)
    solver = mwpf.SolverSerialJointSingleHair(initializer)

    visualizer = mwpf.Visualizer(
        positions=[mwpf.VisualizePosition(0, 0, 0), mwpf.VisualizePosition(1, 0, 0)]
    )
    syndrome = mwpf.SyndromePattern([0])
    try:
        try:
            solver.solve(syndrome, visualizer)  # unsolvable, and should panic

            visualizer.save_html(
                os.path.join(os.path.dirname(__file__), f"test_basic_panic.html")
            )

            solver.subgraph()
        except BaseException as panic:
            raise ValueError(mwpf.panic_text_of(solver, syndrome)) from panic
    except BaseException:
        panic_text = traceback.format_exc()
        # print(panic_text)
        assert "######## MWPF Sinter Decoder Panic ########" in panic_text
        pass
    else:
        assert False, "panic expected"

    # after the panic, the solver should still be able to report the information
    post_panic_initializer = solver.get_initializer()
    assert post_panic_initializer.to_json() == initializer.to_json()
    post_panic_config = solver.config
    print(post_panic_config)
