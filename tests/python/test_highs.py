import importlib
import math
from concurrent.futures import ThreadPoolExecutor
from random import Random

from common import mwpf, pytest

highs = importlib.import_module(f"{mwpf.__name__}._highs")


def distance_model(checks, logicals, count, cap=None):
    """Encode fault parities with integer slack, as qdk.ec's backend does."""
    model = highs.Highs()
    assert model.setOptionValue("output_flag", False) == highs.HighsStatus.kOk
    assert model.setOptionValue("mip_rel_gap", 0.0) == highs.HighsStatus.kOk
    assert model.setOptionValue("mip_abs_gap", 0.0) == highs.HighsStatus.kOk
    faults = [int(model.addBinary(obj=1.0)) for _ in range(count)]
    changed = []
    for logical, rows in ((False, checks), (True, logicals)):
        for support in rows:
            indices = [faults[index] for index in support]
            slack = int(model.addIntegral(lb=0, ub=len(support) // 2))
            indices.append(slack)
            coefficients = [1.0] * len(support) + [-2.0]
            if logical:
                indicator = int(model.addBinary())
                changed.append(indicator)
                indices.append(indicator)
                coefficients.append(-1.0)
            assert (
                model.addRow(0.0, 0.0, len(indices), indices, coefficients)
                == highs.HighsStatus.kOk
            )
    model.addRow(1.0, highs.kHighsInf, len(changed), changed, [1.0] * len(changed))
    if cap is not None:
        model.addRow(0.0, cap, count, faults, [1.0] * count)
    return model, faults


@pytest.mark.parametrize(
    "checks,logicals,count,expected",
    [
        ([[0, 1], [1, 2]], [[0, 1, 2]], 3, 3),  # X distance of repetition-3
        ([[0, 2], [0, 1], [1, 2]], [[0]], 3, 3),  # odd triangle
        ([[0, 3], [0, 1], [1, 2], [2, 3]], [[0]], 4, 4),  # odd square
        ([], [[0], [1]], 2, 1),  # any nonzero logical indicator suffices
    ],
)
def test_distance_parity_milp(checks, logicals, count, expected):
    model, faults = distance_model(checks, logicals, count)
    assert model.run() == highs.HighsStatus.kOk
    assert model.getModelStatus() == highs.HighsModelStatus.kOptimal
    info, solution = model.getInfo(), model.getSolution()
    assert info.valid and solution.value_valid
    witness = [index for index in faults if solution.col_value[index] > 0.5]
    assert len(witness) == expected
    assert info.mip_dual_bound == pytest.approx(expected)
    assert info.objective_function_value == pytest.approx(expected)
    assert all(sum(index in witness for index in row) % 2 == 0 for row in checks)
    assert any(sum(index in witness for index in row) % 2 == 1 for row in logicals)


@pytest.mark.parametrize("cutoff", [False, True])
def test_infeasible_distance_has_no_witness(cutoff):
    if cutoff:
        model, _ = distance_model([[0, 1], [1, 2]], [[0, 1, 2]], 3, cap=2)
    else:
        model, _ = distance_model([[0], [1]], [[0, 1]], 2)
    assert model.run() == highs.HighsStatus.kOk
    assert model.getModelStatus() == highs.HighsModelStatus.kInfeasible
    assert not model.getSolution().value_valid
    assert model.getSolution().col_value == []


def test_timeout_does_not_invent_a_witness():
    model = highs.Highs()
    rng = Random(7)
    for _ in range(12):
        model.addBinary(obj=1)
    for _ in range(24):
        indices = rng.sample(range(12), 5)
        model.addRow(2, highs.kHighsInf, 5, indices, [1.0] * 5)
    model.setOptionValue("time_limit", 0.0)
    assert model.run() in (highs.HighsStatus.kOk, highs.HighsStatus.kWarning)
    assert model.getModelStatus() == highs.HighsModelStatus.kTimeLimit
    assert not model.getSolution().value_valid


def test_mutation_invalidates_and_rerun_uses_new_constraints():
    model = highs.Highs()
    x = model.addIntegral(lb=0, ub=5, obj=-1)
    model.run()
    assert model.getSolution().col_value == [5]
    model.addRow(0, 2, 1, [x], [1])
    assert model.getModelStatus() == highs.HighsModelStatus.kNotset
    assert not model.getInfo().valid
    assert not model.getSolution().value_valid
    model.run()
    assert model.getSolution().col_value == [2]
    detached = model.getSolution()
    detached.col_value[0] = 999
    assert model.getSolution().col_value == [2]


def test_empty_model_is_not_a_valid_solution():
    model = highs.Highs()
    assert model.getModelStatus() == highs.HighsModelStatus.kNotset
    assert not model.getInfo().valid
    model.run()
    assert model.getModelStatus() == highs.HighsModelStatus.kModelEmpty
    assert not model.getInfo().valid
    assert not model.getSolution().value_valid
    assert model.getSolution().col_value == []


def test_independent_milps_and_decoder_can_run_together():
    from test_wheel_compatibility import decode

    def solve(index):
        model, _ = distance_model([[0, 1], [1, 2]], [[0, 1, 2]], 3)
        model.run()
        assert model.getInfo().mip_dual_bound == 3
        return decode(index)

    with ThreadPoolExecutor(max_workers=4) as executor:
        assert all(executor.map(solve, range(16)))


@pytest.mark.parametrize("name", ["time_limit", "mip_rel_gap", "mip_abs_gap"])
@pytest.mark.parametrize("value", [-1, math.nan, math.inf])
def test_invalid_options_are_explicit_errors(name, value):
    with pytest.raises(ValueError):
        highs.Highs().setOptionValue(name, value)


def test_unsupported_option_is_not_silently_ignored():
    with pytest.raises(ValueError, match="unsupported"):
        highs.Highs().setOptionValue("solver", "ipm")
    with pytest.raises(TypeError, match="bool"):
        highs.Highs().setOptionValue("output_flag", "off")


@pytest.mark.parametrize(
    "lower,upper,count,indices,values",
    [
        (0, 1, 2, [0], [1]),
        (0, 1, 1, [1], [1]),
        (0, 1, 1, [-1], [1]),
        (0, 1, 2, [0, 0], [1, 1]),
        (0, 1, 1, [0], [math.nan]),
        (2, 1, 1, [0], [1]),
    ],
)
def test_invalid_sparse_rows_are_rejected(lower, upper, count, indices, values):
    model = highs.Highs()
    model.addBinary()
    with pytest.raises(ValueError):
        model.addRow(lower, upper, count, indices, values)


def test_native_entry_point_also_validates_input():
    with pytest.raises(ValueError, match="lengths"):
        highs.mwpf._highs_solve([1.0], [], [1.0], [], None, False, 0.0, 0.0)
    with pytest.raises(ValueError, match="sparse"):
        highs.mwpf._highs_solve(
            [1.0], [0.0], [1.0], [(0.0, 1.0, [100], [1.0])], None, False, 0.0, 0.0
        )
