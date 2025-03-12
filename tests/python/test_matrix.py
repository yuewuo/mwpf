from common import *


def test_echelon_matrix_simple():
    # pytest -s tests/python/test_matrix.py::test_echelon_matrix_simple
    matrix = mwpf.EchelonMatrix()
    matrix.add_constraint(vertex_index=0, incident_edges=[1, 4, 6], parity=True)
    matrix.add_constraint(1, [4, 9], parity=False)
    matrix.add_constraint(2, [1, 9], parity=True)
    assert matrix.edge_to_var_index(4) == 1
    for edge_index in [1, 4, 6, 9]:
        matrix.update_edge_tightness(edge_index, True)
    print()
    print(matrix)
    assert (
        str(matrix)
        == """\
┌──┬─┬─┬─┬─┬───┬─┐
┊ E┊1┊4┊6┊9┊ = ┊▼┊
╞══╪═╪═╪═╪═╪═══╪═╡
┊ 0┊1┊ ┊ ┊1┊ 1 ┊1┊
├──┼─┼─┼─┼─┼───┼─┤
┊ 1┊ ┊1┊ ┊1┊   ┊4┊
├──┼─┼─┼─┼─┼───┼─┤
┊ 2┊ ┊ ┊1┊ ┊   ┊6┊
├──┼─┼─┼─┼─┼───┼─┤
┊ ▶┊0┊1┊2┊*┊◀  ┊▲┊
└──┴─┴─┴─┴─┴───┴─┘
"""
    )
    # set edges 1 and 6 as the tail
    matrix.set_tail_edges({6, 1})
    assert matrix.get_tail_edges() == {1, 6}
    print(matrix)
    assert (
        str(matrix)
        == """\
┌──┬─┬─┬─┬─┬───┬─┐
┊ E┊4┊9┊1┊6┊ = ┊▼┊
╞══╪═╪═╪═╪═╪═══╪═╡
┊ 0┊1┊ ┊1┊ ┊ 1 ┊4┊
├──┼─┼─┼─┼─┼───┼─┤
┊ 1┊ ┊1┊1┊ ┊ 1 ┊9┊
├──┼─┼─┼─┼─┼───┼─┤
┊ 2┊ ┊ ┊ ┊1┊   ┊6┊
├──┼─┼─┼─┼─┼───┼─┤
┊ ▶┊0┊1┊*┊2┊◀  ┊▲┊
└──┴─┴─┴─┴─┴───┴─┘
"""
    )
    # set edge 4 as the tail
    matrix.set_tail_edges({4})
    print(matrix)
    assert (
        str(matrix)
        == """\
┌──┬─┬─┬─┬─┬───┬─┐
┊ E┊1┊6┊9┊4┊ = ┊▼┊
╞══╪═╪═╪═╪═╪═══╪═╡
┊ 0┊1┊ ┊ ┊1┊ 1 ┊1┊
├──┼─┼─┼─┼─┼───┼─┤
┊ 1┊ ┊1┊ ┊ ┊   ┊6┊
├──┼─┼─┼─┼─┼───┼─┤
┊ 2┊ ┊ ┊1┊1┊   ┊9┊
├──┼─┼─┼─┼─┼───┼─┤
┊ ▶┊0┊1┊2┊*┊◀  ┊▲┊
└──┴─┴─┴─┴─┴───┴─┘
"""
    )
    # set one edge to loose
    matrix.update_edge_tightness(edge_index=6, is_tight=False)
    print(matrix)
    assert (
        str(matrix)
        == """\
┌──┬─┬─┬─┬───┬─┐
┊ E┊1┊9┊4┊ = ┊▼┊
╞══╪═╪═╪═╪═══╪═╡
┊ 0┊1┊ ┊1┊ 1 ┊1┊
├──┼─┼─┼─┼───┼─┤
┊ 1┊ ┊1┊1┊   ┊9┊
├──┼─┼─┼─┼───┼─┤
┊ ▶┊0┊1┊*┊◀  ┊▲┊
└──┴─┴─┴─┴───┴─┘
"""
    )
    matrix.update_edge_tightness(1, False)
    matrix.update_edge_tightness(9, False)
    print(matrix)
    assert (
        str(matrix)
        == """\
┌──┬─┬───┬─┐
┊ X┊4┊ = ┊▼┊
╞══╪═╪═══╪═╡
┊ 0┊1┊ 1 ┊4┊
├──┼─┼───┼─┤
┊ 1┊ ┊ 1 ┊*┊
├──┼─┼───┼─┤
┊ ▶┊0┊◀  ┊▲┊
└──┴─┴───┴─┘
"""
    )


def test_tail_matrix_1():
    # pytest -s tests/python/test_matrix.py::test_tail_matrix_1
    matrix = mwpf.TailMatrix()
    matrix.add_constraint(vertex_index=0, incident_edges=[1, 4, 6], parity=True)
    matrix.add_constraint(1, [4, 9], parity=False)
    matrix.add_constraint(2, [1, 9], parity=True)
    # assert matrix.edge_to_var_index(4) == 1
    print()
    print(matrix)
    assert (
        str(matrix)
        == """\
┌─┬───┐
┊ ┊ = ┊
╞═╪═══╡
┊0┊ 1 ┊
├─┼───┤
┊1┊   ┊
├─┼───┤
┊2┊ 1 ┊
└─┴───┘
"""
    )
    for edge_index in [1, 4, 6, 9]:
        matrix.update_edge_tightness(edge_index, True)
    print(matrix)
    assert (
        str(matrix)
        == """\
┌─┬─┬─┬─┬─┬───┐
┊ ┊1┊4┊6┊9┊ = ┊
╞═╪═╪═╪═╪═╪═══╡
┊0┊1┊1┊1┊ ┊ 1 ┊
├─┼─┼─┼─┼─┼───┤
┊1┊ ┊1┊ ┊1┊   ┊
├─┼─┼─┼─┼─┼───┤
┊2┊1┊ ┊ ┊1┊ 1 ┊
└─┴─┴─┴─┴─┴───┘
"""
    )
    # set edges 1 and 6 as the tail
    matrix.set_tail_edges({6, 1})
    assert matrix.get_tail_edges() == {1, 6}
    print(matrix)
    assert (
        str(matrix)
        == """\
┌─┬─┬─┬─┬─┬───┐
┊ ┊4┊9┊1┊6┊ = ┊
╞═╪═╪═╪═╪═╪═══╡
┊0┊1┊ ┊1┊1┊ 1 ┊
├─┼─┼─┼─┼─┼───┤
┊1┊1┊1┊ ┊ ┊   ┊
├─┼─┼─┼─┼─┼───┤
┊2┊ ┊1┊1┊ ┊ 1 ┊
└─┴─┴─┴─┴─┴───┘
"""
    )


def test_tight_matrix_1():
    # pytest -s tests/python/test_matrix.py::test_tight_matrix_1
    matrix = mwpf.TightMatrix()
    matrix.add_constraint(vertex_index=0, incident_edges=[1, 4, 6], parity=True)
    matrix.add_constraint(1, [4, 9], parity=False)
    matrix.add_constraint(2, [1, 9], parity=True)
    assert matrix.edge_to_var_index(4) == 1
    print()
    print(matrix)
    assert (
        str(matrix)
        == """\
┌─┬───┐
┊ ┊ = ┊
╞═╪═══╡
┊0┊ 1 ┊
├─┼───┤
┊1┊   ┊
├─┼───┤
┊2┊ 1 ┊
└─┴───┘
"""
    )
    matrix.update_edge_tightness(4, True)
    matrix.update_edge_tightness(9, True)
    print(matrix)
    assert (
        str(matrix)
        == """\
┌─┬─┬─┬───┐
┊ ┊4┊9┊ = ┊
╞═╪═╪═╪═══╡
┊0┊1┊ ┊ 1 ┊
├─┼─┼─┼───┤
┊1┊1┊1┊   ┊
├─┼─┼─┼───┤
┊2┊ ┊1┊ 1 ┊
└─┴─┴─┴───┘
"""
    )
    matrix.update_edge_tightness(9, False)
    print(matrix)
    assert (
        str(matrix)
        == """\
┌─┬─┬───┐
┊ ┊4┊ = ┊
╞═╪═╪═══╡
┊0┊1┊ 1 ┊
├─┼─┼───┤
┊1┊1┊   ┊
├─┼─┼───┤
┊2┊ ┊ 1 ┊
└─┴─┴───┘
"""
    )


def test_basic_matrix_1():
    # pytest -s tests/python/test_matrix.py::test_basic_matrix_1
    matrix = mwpf.BasicMatrix()
    print()
    print(matrix)
    assert (
        str(matrix)
        == """\
┌┬───┐
┊┊ = ┊
╞╪═══╡
└┴───┘
"""
    )
    matrix.add_variable(edge_index=1)
    matrix.add_variable(4)
    matrix.add_variable(12)
    matrix.add_variable(345)
    print(matrix)
    assert (
        str(matrix)
        == """\
┌┬─┬─┬─┬─┬───┐
┊┊1┊4┊1┊3┊ = ┊
┊┊ ┊ ┊2┊4┊   ┊
┊┊ ┊ ┊ ┊5┊   ┊
╞╪═╪═╪═╪═╪═══╡
└┴─┴─┴─┴─┴───┘
"""
    )
    matrix.add_constraint(0, [1, 4, 12], True)
    matrix.add_constraint(1, [4, 345], False)
    matrix.add_constraint(2, [1, 345], True)
    print(matrix)
    assert (
        str(matrix)
        == """\
┌─┬─┬─┬─┬─┬───┐
┊ ┊1┊4┊1┊3┊ = ┊
┊ ┊ ┊ ┊2┊4┊   ┊
┊ ┊ ┊ ┊ ┊5┊   ┊
╞═╪═╪═╪═╪═╪═══╡
┊0┊1┊1┊1┊ ┊ 1 ┊
├─┼─┼─┼─┼─┼───┤
┊1┊ ┊1┊ ┊1┊   ┊
├─┼─┼─┼─┼─┼───┤
┊2┊1┊ ┊ ┊1┊ 1 ┊
└─┴─┴─┴─┴─┴───┘
"""
    )
    assert matrix.get_vertices() == {0, 1, 2}
    assert matrix.get_view_edges() == [1, 4, 12, 345]


def test_matrix_from_cluster():
    # pytest -s tests/python/test_matrix.py::test_matrix_from_cluster
    code = mwpf.CodeCapacityColorCode(d=5, p=0.005)
    visualizer = mwpf.Visualizer(positions=code.get_positions())
    solver = mwpf.Solver(code.get_initializer().uniform_weights())
    solver.solve(mwpf.SyndromePattern([2, 3, 7]), visualizer)
    visualizer.save_html(
        os.path.join(os.path.dirname(__file__), f"test_matrix_from_cluster.html")
    )
    # extract the parity matrix of the cluster
    vertex_index = 2  # choose an arbitrary vertex in the cluster (use the vis tool)
    cluster = solver.get_cluster(vertex_index)
    print(cluster)
    # check the cluster
    assert cluster.vertices == {2, 3, 7}
    assert cluster.edges == {10}
    assert cluster.hair == {9, 5, 6, 2, 3, 7, 11, 16, 15, 13, 12, 14}
    assert cluster.nodes == {solver.get_node(0), solver.get_node(1), solver.get_node(2)}
    print()
    print(cluster.parity_matrix)
    assert (
        str(cluster.parity_matrix)
        == """\
┌─┬─┬───┐
┊ ┊1┊ = ┊
┊ ┊0┊   ┊
╞═╪═╪═══╡
┊0┊1┊ 1 ┊
├─┼─┼───┤
┊1┊1┊ 1 ┊
├─┼─┼───┤
┊2┊1┊ 1 ┊
└─┴─┴───┘
"""
    )
    print(mwpf.BasicMatrix(cluster.parity_matrix))
    # because the order of the cluster isn't fixed, we cannot assert a single matrix printout
    #     assert (
    #         str(mwpf.BasicMatrix(cluster.parity_matrix))
    #         == """\
    # ┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬───┐
    # ┊ ┊1┊5┊6┊9┊1┊1┊2┊3┊7┊1┊1┊1┊1┊ = ┊
    # ┊ ┊0┊ ┊ ┊ ┊2┊3┊ ┊ ┊ ┊1┊4┊5┊6┊   ┊
    # ╞═╪═╪═╪═╪═╪═╪═╪═╪═╪═╪═╪═╪═╪═╪═══╡
    # ┊0┊1┊1┊1┊1┊1┊1┊ ┊ ┊ ┊ ┊ ┊ ┊ ┊ 1 ┊
    # ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼───┤
    # ┊1┊1┊ ┊1┊ ┊ ┊ ┊1┊1┊1┊1┊ ┊ ┊ ┊ 1 ┊
    # ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼───┤
    # ┊2┊1┊ ┊ ┊ ┊ ┊1┊ ┊ ┊ ┊1┊1┊1┊1┊ 1 ┊
    # └─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴───┘
    # """)
    basic_matrix = mwpf.BasicMatrix(cluster.parity_matrix)
    assert basic_matrix.rows == 3
    assert basic_matrix.columns == 13
    rows = set()
    for row in range(3):
        edges = set()
        for column in range(13):
            if basic_matrix.get_lhs(row, column):
                edges.add(basic_matrix.column_to_edge_index(column))
        rows.add(frozenset(edges))
        assert basic_matrix.get_rhs(row) == True
    assert rows == {
        frozenset({5, 6, 9, 10, 12, 13}),
        frozenset({2, 3, 6, 7, 10, 11}),
        frozenset({16, 10, 11, 13, 14, 15}),
    }
    # rows[row[0]] = row[1:]
