"""Internal, minimization-only highspy subset for QDK distance MILPs.

This is not a general replacement for highspy. It supports integer variables,
sparse rows, and the options/results used by qdk.ec's distance backend.
Independent model instances can run concurrently; do not share a mutable model
between threads.
"""

import math
import operator
from dataclasses import dataclass, field
from enum import IntEnum

from . import mwpf

kHighsInf = float("inf")


class HighsStatus(IntEnum):
    kError = -1
    kOk = 0
    kWarning = 1


class HighsModelStatus(IntEnum):
    kNotset = 0
    kLoadError = 1
    kModelError = 2
    kPresolveError = 3
    kSolveError = 4
    kPostsolveError = 5
    kModelEmpty = 6
    kOptimal = 7
    kInfeasible = 8
    kUnboundedOrInfeasible = 9
    kUnbounded = 10
    kObjectiveBound = 11
    kObjectiveTarget = 12
    kTimeLimit = 13
    kIterationLimit = 14
    kUnknown = 15
    kSolutionLimit = 16
    kInterrupt = 17


@dataclass
class HighsInfo:
    valid: bool = False
    mip_dual_bound: float = math.nan
    objective_function_value: float = math.nan


@dataclass
class HighsSolution:
    value_valid: bool = False
    col_value: list[float] = field(default_factory=list)


def _finite(value, name):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _bounds(lower, upper):
    lower, upper = float(lower), float(upper)
    if (
        math.isnan(lower)
        or math.isnan(upper)
        or lower > upper
        or lower == math.inf
        or upper == -math.inf
    ):
        raise ValueError("invalid HiGHS bounds")
    return lower, upper


class Highs:
    def __init__(self):
        self._costs = []
        self._lower = []
        self._upper = []
        self._rows = []
        self._options = {
            "output_flag": False,
            "time_limit": None,
            "mip_rel_gap": 0.0,
            "mip_abs_gap": 0.0,
        }
        self._invalidate()

    def _invalidate(self):
        self._status = HighsModelStatus.kNotset
        self._info = HighsInfo()
        self._solution = HighsSolution()

    def setOptionValue(self, name, value):
        if name not in self._options:
            raise ValueError(f"unsupported HiGHS option: {name}")
        if name == "output_flag":
            if not isinstance(value, bool):
                raise TypeError("output_flag must be a bool")
        else:
            value = _finite(value, name)
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        self._options[name] = value
        self._invalidate()
        return HighsStatus.kOk

    def addBinary(self, obj=0.0):
        return self.addIntegral(lb=0.0, ub=1.0, obj=obj)

    def addIntegral(self, lb=0.0, ub=kHighsInf, obj=0.0):
        lower, upper = _bounds(lb, ub)
        cost = _finite(obj, "objective coefficient")
        index = len(self._costs)
        self._costs.append(cost)
        self._lower.append(lower)
        self._upper.append(upper)
        self._invalidate()
        return index

    def addRow(self, lower, upper, num_new_nz, indices, values):
        lower, upper = _bounds(lower, upper)
        count = operator.index(num_new_nz)
        indices = [operator.index(index) for index in indices]
        values = [_finite(value, "row coefficient") for value in values]
        if count != len(indices) or count != len(values):
            raise ValueError("sparse row lengths must match num_new_nz")
        if any(index < 0 or index >= len(self._costs) for index in indices):
            raise ValueError("column index out of range")
        if len(set(indices)) != len(indices):
            raise ValueError("duplicate column indices in row")
        self._rows.append((lower, upper, indices, values))
        self._invalidate()
        return HighsStatus.kOk

    def run(self):
        self._invalidate()
        status, model_status, info_valid, value_valid, bound, objective, values = (
            mwpf._highs_solve(
                self._costs,
                self._lower,
                self._upper,
                self._rows,
                self._options["time_limit"],
                self._options["output_flag"],
                self._options["mip_rel_gap"],
                self._options["mip_abs_gap"],
            )
        )
        self._status = HighsModelStatus(model_status)
        self._info = HighsInfo(info_valid, bound, objective)
        self._solution = HighsSolution(value_valid, values)
        return HighsStatus(status)

    def getModelStatus(self):
        return self._status

    def getInfo(self):
        return HighsInfo(
            self._info.valid,
            self._info.mip_dual_bound,
            self._info.objective_function_value,
        )

    def getSolution(self):
        return HighsSolution(self._solution.value_valid, self._solution.col_value.copy())

    def modelStatusToString(self, status):
        return HighsModelStatus(status).name[1:]
