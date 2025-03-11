"""
The Circuit object in stim always write in relative measurement index, 
which is great for writing loops but not so great for analyzing the measurements.
Especially, if we want to analyze the effect of certain heralded errors, we will
need to keep the rest of the measurements in the same place. However, removing or 
adding one heralded error will change all the relative measurement indices, making it
especially hard to track and analyze the circuit.

This class is used to convert the relative measurement index to referenced measurement
 and vice versa. This will allow us to manipulate the circuit by adding or removing
heralded errors more easily.

This is done by adding a custom class `RefRec` in place of the normal `GateTarget`
object returned by the `stim.target_rec(lookback)` function.
In contrast to the negative integer `lookback` parameters, the `bias` integer is a non-negative
index bias over the `RefInstruction` that generates the measurements.
In this way, we are free to manipulate the instructions without worrying that the measurements
indices are messed up.
The `RefCircuit` object will automatically calculate the new relative indices so that the circuit
works properly, and also it will generate a mapping between indices of the old circuit and new circuit.

```python
circuit = stim.Circuit(...)
ref_circuit = RefCircuit.of(circuit)
print(ref_circuit)  # print the circuit in absolute indices
circuit_2 = ref_circuit.to_circuit()  # convert the ref_circuit back to stim.Circuit
print(circuit_2)  # print the circuit in relative indices
```
"""

import stim
from dataclasses import dataclass, field
from typing import Iterator, Iterable, TypeAlias, Collection, Protocol, Sequence, Any
import functools
import numpy as np
from functools import reduce
from frozendict import frozendict
from frozenlist import FrozenList
import mwpf


@dataclass(frozen=True)
class RefRec:
    instruction: "RefInstruction"
    bias: int

    def __post_init__(self):
        if self.bias < 0:
            raise ValueError("The bias must be a non-negative integer.")

    def abs_index(self, circuit: "RefCircuit") -> int:
        return circuit.rec_to_index[self]

    def rel_index(self, circuit: "RefCircuit", instruction: "RefInstruction") -> int:
        """get the relative rec index before executing the instruction"""
        return circuit.rec_to_index[self] - circuit.instruction_to_rec_bias[instruction]

    def __eq__(self, other: object) -> bool:
        return self is other  # avoiding value-based comparison

    def __hash__(self) -> int:
        return hash(id(self))


@dataclass(frozen=True)
class RefInstruction:
    name: str
    targets: tuple[stim.GateTarget | RefRec, ...] = ()
    gate_args: tuple[float, ...] = ()
    tag: str = ""
    recs: FrozenList[RefRec] = field(default_factory=FrozenList)

    @property
    def num_measurements(self) -> int:
        return len(self.recs)

    def index(self, circuit: "RefCircuit") -> int:
        return circuit.instruction_to_index[self]

    def __eq__(self, other: object) -> bool:
        return self is other  # avoiding value-based comparison

    def __hash__(self) -> int:
        return hash(id(self))

    @staticmethod
    def new_heralded_erase(
        targets: tuple[stim.GateTarget | RefRec, ...], p: float
    ) -> "RefInstruction":
        instruction = RefInstruction(
            name="HERALDED_ERASE",
            targets=targets,
            gate_args=(p,),
        )
        for bias, target in enumerate(targets):
            assert isinstance(target, stim.GateTarget)
            instruction.recs.append(
                RefRec(
                    instruction=instruction,
                    bias=bias,
                )
            )
        instruction.recs.freeze()  # do not allow further edit
        return instruction

    @staticmethod
    def new_heralded_pauli_channel_1(
        targets: tuple[stim.GateTarget | RefRec, ...],
        pI: float = 0,
        pX: float = 0,
        pY: float = 0,
        pZ: float = 0,
    ) -> "RefInstruction":
        instruction = RefInstruction(
            name="HERALDED_PAULI_CHANNEL_1",
            targets=targets,
            gate_args=(pI, pX, pY, pZ),
        )
        for bias, target in enumerate(targets):
            assert isinstance(target, stim.GateTarget)
            instruction.recs.append(
                RefRec(
                    instruction=instruction,
                    bias=bias,
                )
            )
        instruction.recs.freeze()  # do not allow further edit
        return instruction

    @staticmethod
    def new_detector(targets: tuple[stim.GateTarget | RefRec, ...]) -> "RefInstruction":
        for target in targets:
            assert isinstance(target, RefRec)
        return RefInstruction(
            name="DETECTOR",
            targets=targets,
        )


RefDetector: TypeAlias = RefInstruction


@dataclass(frozen=True)
class RefCircuit:
    instructions: tuple[RefInstruction, ...]

    def __post_init__(self):
        self.sanity_check()

    @staticmethod
    def of(
        circuit: stim.Circuit | Iterable[RefInstruction] | None = None,
    ) -> "RefCircuit":
        if circuit is None:
            return RefCircuit(instructions=tuple())
        if not isinstance(circuit, stim.Circuit):
            return RefCircuit(instructions=tuple(circuit))
        instructions: list[RefInstruction] = []
        recs: list[RefRec] = []
        for instruction in circuit.flattened():
            # rewrite the gate target to absolute measurement indices
            ref_targets: list[stim.GateTarget | RefRec] = []
            for target in instruction.targets_copy():
                if target.is_measurement_record_target:
                    ref_targets.append(recs[target.value])
                else:
                    ref_targets.append(target)
            # construct the instruction
            ref_instruction = RefInstruction(
                name=instruction.name,
                targets=tuple(ref_targets),
                gate_args=tuple(instruction.gate_args_copy()),
                **extra_instruction_params(instruction),
            )
            instructions.append(ref_instruction)
            # add the measurement to the measurement list
            for bias in range(instruction.num_measurements):
                reference_rec = RefRec(
                    instruction=ref_instruction,
                    bias=bias,
                )
                ref_instruction.recs.append(reference_rec)
                recs.append(reference_rec)
            ref_instruction.recs.freeze()  # do not allow further edit
        return RefCircuit(instructions=tuple(instructions))

    def __repr__(self) -> str:
        repr_str = "RefCircuit: {  # not executable\n"
        repr_str += "\n".join("    " + e for e in str(self).splitlines())
        repr_str += "\n}"
        return repr_str

    def __str__(self) -> str:
        # print the circuit in absolute indices; this helps writing test cases
        code_str = ""
        for instruction, stim_instruction in zip(
            self.instructions, self.stim_instructions
        ):
            rec_count: dict[RefRec, int] = {}
            for ref_target in instruction.targets:
                if isinstance(ref_target, RefRec):
                    if ref_target in rec_count:
                        rec_count[ref_target] += 1
                    else:
                        rec_count[ref_target] = 1
            instruction_str = str(stim_instruction)
            # change all the relative indices to absolute indices
            for ref_rec, count in rec_count.items():
                # this count checking is safe because "rec[..]" becomes "rec[..\C"
                # in the tag parameter, and the character "]" cannot appear elsewhere
                relative_index = ref_rec.rel_index(self, instruction)
                assert instruction_str.count(f"rec[{relative_index}]") == count
                instruction_str = instruction_str.replace(
                    f"rec[{relative_index}]", f"abs[{ref_rec.abs_index(self)}]"
                )
            if len(code_str) > 0:
                code_str += "\n"
            code_str += instruction_str
        return code_str

    def __iter__(self) -> Iterator[RefInstruction]:
        for instruction in self.instructions:
            yield instruction

    @functools.cached_property
    def recs(self) -> tuple[RefRec, ...]:
        recs: list[RefRec] = []
        for instruction in self.instructions:
            recs.extend(instruction.recs)
        return tuple(recs)

    @functools.cached_property
    def rec_to_index(self) -> frozendict[RefRec, int]:
        return frozendict({ref_rec: index for index, ref_rec in enumerate(self.recs)})

    @functools.cached_property
    def rec_to_detectors(self) -> frozendict[RefRec, tuple[RefDetector, ...]]:
        rec_to_detectors: dict[RefRec, list[RefDetector]] = {
            rec: [] for rec in self.recs
        }
        for detector in self.detectors:
            for ref_target in detector.targets:
                assert isinstance(ref_target, RefRec)
                rec_to_detectors[ref_target].append(detector)
        return frozendict(
            {
                ref_rec: tuple(detectors)
                for ref_rec, detectors in rec_to_detectors.items()
            }
        )

    @functools.cached_property
    def instruction_to_index(self) -> frozendict[RefInstruction, int]:
        return frozendict(
            {instruction: index for index, instruction in enumerate(self.instructions)}
        )

    @functools.cached_property
    def instruction_rec_biases(self) -> tuple[int, ...]:
        rec_bias: int = 0
        rec_biases: list[int] = []
        for instruction in self.instructions:
            rec_biases.append(rec_bias)
            rec_bias += instruction.num_measurements
        return tuple(rec_biases)

    @functools.cached_property
    def instruction_to_rec_bias(self) -> frozendict[RefInstruction, int]:
        return frozendict(
            {
                instruction: bias
                for instruction, bias in zip(
                    self.instructions, self.instruction_rec_biases
                )
            }
        )

    @functools.cached_property
    def detectors(self) -> tuple[RefDetector, ...]:
        detectors: list[RefDetector] = []
        for instruction in self.instructions:
            if instruction.name == "DETECTOR":
                detectors.append(instruction)
        return tuple(detectors)

    @functools.cached_property
    def detector_to_index(self) -> frozendict[RefDetector, int]:
        return frozendict(
            {detector: index for index, detector in enumerate(self.detectors)}
        )

    @functools.cached_property
    def stim_instructions(self) -> tuple[stim.CircuitInstruction, ...]:
        stim_instructions: list[stim.CircuitInstruction] = []
        for instruction in self.instructions:
            relative_targets = []
            for ref_target in instruction.targets:
                if isinstance(ref_target, RefRec):
                    relative_targets.append(
                        stim.target_rec(ref_target.rel_index(self, instruction))
                    )
                else:
                    relative_targets.append(ref_target)
            stim_instructions.append(
                stim.CircuitInstruction(
                    name=instruction.name,
                    targets=relative_targets,
                    gate_args=instruction.gate_args,
                    **extra_instruction_params(instruction),
                )
            )
        return tuple(stim_instructions)

    @functools.cached_property
    def ref_dem(self) -> "RefDetectorErrorModel":
        return RefDetectorErrorModel.of(self)

    def circuit(self) -> stim.Circuit:
        circuit = stim.Circuit()
        for stim_instruction in self.stim_instructions:
            circuit.append(stim_instruction)
        return circuit

    def __getitem__(self, index_or_slice: object) -> "RefInstruction | RefCircuit":
        if isinstance(index_or_slice, int):
            return self.instructions[index_or_slice]
        elif isinstance(index_or_slice, slice):
            return RefCircuit.of(self.instructions[index_or_slice])
        else:
            raise TypeError("Invalid key type")

    def __add__(self, other: "RefCircuit | RefInstruction") -> "RefCircuit":
        if isinstance(other, RefInstruction):
            return RefCircuit.of([*self.instructions, other])
        elif isinstance(other, RefCircuit):
            return RefCircuit.of([*self.instructions, *other.instructions])
        else:
            raise NotImplemented()

    def __radd__(self, other: "RefCircuit | RefInstruction") -> "RefCircuit":
        if isinstance(other, RefInstruction):
            return RefCircuit.of([other, *self.instructions])
        elif isinstance(other, RefCircuit):
            return RefCircuit.of([*other.instructions, *self.instructions])
        else:
            raise NotImplemented()

    def sanity_check(self) -> "RefCircuit":
        # check basic type
        assert isinstance(self.instructions, tuple)
        for instruction in self.instructions:
            assert isinstance(instruction, RefInstruction)
        # check if there are any duplicate instruction: id -> index
        existing_instruction: dict[RefInstruction, int] = {}
        for index, instruction in enumerate(self.instructions):
            assert instruction not in existing_instruction, (
                f"Duplicate instruction found instruction={instruction}, "
                + f"previous index={existing_instruction[instruction]}, "
                + f"current index={index}; if you want to repeat the same "
                + "circuit, considering using sub_circuit + sub_circuit.clone()"
            )
            existing_instruction[instruction] = index
        # check if all the references are valid in the current context
        rec_index_of: dict[RefRec, int] = {
            ref_rec: index for index, ref_rec in enumerate(self.recs)
        }
        assert len(rec_index_of) == len(self.recs), "has duplicate RefRec object"
        rec_hashes = set(hash(rec) for rec in self.recs)
        assert len(rec_hashes) == len(self.recs), "hash conflict"
        # check if the instructions only reference to the available RefRec objects
        # also check that all the RefRec objects refer to a valid Instruction
        previous_rec_set: set[RefRec] = set()
        previous_instruction_set: set[RefInstruction] = set()
        for instruction in self.instructions:
            for ref_target in instruction.targets:
                if isinstance(ref_target, RefRec):
                    assert ref_target in previous_rec_set, (
                        f"Reference to a RefRec object that does not appear previously, "
                        + f"ref_target={ref_target}"
                    )
                    assert ref_target.instruction in previous_instruction_set
            for ref_rec in instruction.recs:
                assert ref_rec.instruction == instruction  # must refer itself
                previous_rec_set.add(ref_rec)
            previous_instruction_set.add(instruction)
        # check that the number of measurements is correct
        for instruction, stim_instruction in zip(
            self.instructions, self.stim_instructions
        ):
            assert stim_instruction.num_measurements == instruction.num_measurements
        # check that all the detectors only targets RefRec objects
        for detector in self.detectors:
            for ref_target in detector.targets:
                assert isinstance(ref_target, RefRec)
        return self

    def clone(self) -> "RefCircuit":
        """
        make a deep copy of the RefCircuit object, creating branch new instructions
        with different ids, but keeping the circuit the same
        """
        new_instructions: list[RefInstruction] = []
        new_rec: list[RefRec] = []
        for instruction in self.instructions:
            new_instruction = RefInstruction(
                name=instruction.name,
                targets=tuple(
                    (
                        new_rec[self.rec_to_index[target]]
                        if isinstance(target, RefRec)
                        else target
                    )
                    for target in instruction.targets
                ),
                gate_args=instruction.gate_args,
                **extra_instruction_params(instruction),
            )
            for bias in range(instruction.num_measurements):
                reference_rec = RefRec(
                    instruction=new_instruction,
                    bias=bias,
                )
                new_instruction.recs.append(reference_rec)
                new_rec.append(reference_rec)
            new_instruction.recs.freeze()  # do not allow further edit
            new_instructions.append(new_instruction)
        return RefCircuit.of(new_instructions)

    def remove_noise_channels(
        self, keeping: Collection[RefInstruction]
    ) -> "RefCircuit":
        """
        Remove all the noise channels and their detectors from the circuit
        """
        new_instructions = list(self)
        deleting_indices: list[int] = []
        for index, instruction in enumerate(self.instructions):
            if not is_noise_channel_instruction(instruction):
                continue
            if instruction not in keeping:
                deleting_indices.append(index)
            # if the noise has measurement result, then they are heralded errors
            # remove the detectors associated with the heralded errors as well
            for ref_rec in instruction.recs:
                for detector in self.rec_to_detectors[ref_rec]:
                    assert (
                        len(detector.targets) == 1
                    ), "bug: detector of a heralded error has multiple targets"
                    if detector not in keeping:
                        deleting_indices.append(self.instruction_to_index[detector])
        assert len(set(deleting_indices)) == len(deleting_indices), "bug: duplicate"
        for index in sorted(deleting_indices, reverse=True):
            del new_instructions[index]
        return RefCircuit.of(new_instructions)


NOISE_CHANNEL_INSTRUCTION_WARNING_KEYWORDS: tuple[str, ...] = (
    "ERROR",
    "DEPOLARIZE",
    "HERALDED",
    "PAULI_CHANNEL",
)

# the available noise channels as of stim v1.15.0-dev
NOISE_CHANNEL_INSTRUCTION_NAMES: frozenset[str] = frozenset(
    {
        "CORRELATED_ERROR",
        "DEPOLARIZE1",
        "DEPOLARIZE2",
        "E",
        "ELSE_CORRELATED_ERROR",
        "HERALDED_ERASE",
        "HERALDED_PAULI_CHANNEL_1",
        "PAULI_CHANNEL_1",
        "PAULI_CHANNEL_2",
        "X_ERROR",
        "Y_ERROR",
        "Z_ERROR",
    }
)


def is_noise_channel_instruction(
    instruction: RefInstruction | stim.CircuitInstruction,
) -> bool:
    name: str = instruction.name
    if name in NOISE_CHANNEL_INSTRUCTION_NAMES:
        return True
    for keyword in NOISE_CHANNEL_INSTRUCTION_WARNING_KEYWORDS:
        if keyword in name:
            print(f"[warning] found keyword {keyword} in unknown instruction: {name}")
    return False


@dataclass(frozen=True)
class RefDemInstruction:
    type: str
    args: tuple[float, ...] = ()
    targets: tuple[int | stim.DemTarget | RefDetector, ...] = ()


@dataclass(frozen=True)
class RefDetectorErrorModel:
    instructions: tuple[RefDemInstruction, ...]
    ref_circuit: RefCircuit

    @staticmethod
    def of(
        ref_circuit: RefCircuit | None = None,
        dem: stim.DetectorErrorModel | None = None,
    ) -> "RefDetectorErrorModel":
        if dem is None:
            assert ref_circuit is not None, "circuit and dem cannot be both None"
            dem = ref_circuit.circuit().detector_error_model(
                approximate_disjoint_errors=True, flatten_loops=True
            )
        if ref_circuit is None:
            # create a mock ref_circuit just for the reference of detectors
            num_detectors = dem.num_detectors
            circuit = stim.Circuit()
            circuit.append("R", list(range(num_detectors)))
            circuit.append(
                "M", list(range(num_detectors))
            )  # some constant measurements
            for detector_id in range(num_detectors):
                circuit.append(
                    "DETECTOR", [stim.target_rec(detector_id - num_detectors)]
                )
            ref_circuit = RefCircuit.of(circuit)
        instructions: list[RefDemInstruction] = []
        for instruction in dem.flattened():
            ref_targets: list[int | stim.DemTarget | RefDetector] = []
            for target in instruction.targets_copy():
                if target.is_relative_detector_id():
                    ref_targets.append(ref_circuit.detectors[target.val])
                else:
                    ref_targets.append(target)
            # construct the instruction
            ref_instruction = RefDemInstruction(
                type=instruction.type,
                args=tuple(instruction.args_copy()),
                targets=tuple(ref_targets),
            )
            instructions.append(ref_instruction)
        return RefDetectorErrorModel(
            instructions=tuple(instructions), ref_circuit=ref_circuit
        )

    def dem(self, ref_circuit: RefCircuit | None = None) -> stim.DetectorErrorModel:
        if ref_circuit is None:
            ref_circuit = self.ref_circuit
        dem = stim.DetectorErrorModel()
        for instruction in self.instructions:
            dem.append(
                stim.DemInstruction(
                    type=instruction.type,
                    args=instruction.args,
                    targets=tuple(
                        (
                            stim.DemTarget.relative_detector_id(
                                ref_circuit.detector_to_index[target]
                            )
                            if isinstance(target, RefDetector)
                            else target
                        )
                        for target in instruction.targets
                    ),
                )
            )
        return dem

    @functools.cached_property
    def hyperedges(self) -> tuple["DemHyperedge", ...]:
        """
        we don't need to put all the hyperedges in the graph. If multiple hyperedges have
        the same detector set (incident vertices) but only differ by logical observable,
        then we only need to add the hyperedge with the highest probability and totally
        ignore others, as they will never be chosen in any MWPF solution anyway.
        """
        # detectors: dict[observables, probability]
        duplicate_edges: dict[frozenset[int], dict[frozenset[int], float]] = {}
        for instruction in self.instructions:
            if instruction.type == "error":
                assert (
                    len(instruction.args) == 1
                ), "error instruction must have 1 parameter of type float"
                probability = instruction.args[0]
                detectors: set[int] = set()
                observables: set[int] = set()
                for target in instruction.targets:
                    if isinstance(target, RefDetector):
                        detectors ^= {self.ref_circuit.detector_to_index[target]}
                    elif (
                        isinstance(target, stim.DemTarget)
                        and target.is_logical_observable_id()
                    ):
                        observables ^= {target.val}
                detectors = frozenset(detectors)
                observables = frozenset(observables)
                if detectors in duplicate_edges:
                    if observables in duplicate_edges[detectors]:
                        old_p = duplicate_edges[detectors][observables]
                        duplicate_edges[detectors][observables] = exclusive_probability(
                            old_p, probability
                        )
                    else:
                        duplicate_edges[detectors][observables] = probability
                else:
                    duplicate_edges[detectors] = {observables: probability}

        # detectors: (probability, observables)
        mapping: dict[frozenset[int], tuple[float, frozenset[int]]] = {}
        for detectors, values in duplicate_edges.items():
            # choosing the observables with the maximum probability
            observables, probability = max(values.items(), key=lambda x: x[1])
            mapping[detectors] = (probability, observables)
        return tuple(
            DemHyperedge(detectors, observables, probability)
            for detectors, (probability, observables) in mapping.items()
        )

    @functools.cached_property
    def hyperedges_detectors_set(self) -> frozenset[frozenset[int]]:
        return frozenset(dem_hyperedge.detectors for dem_hyperedge in self.hyperedges)

    @functools.cached_property
    def _dem(self) -> stim.DetectorErrorModel:
        return self.dem()

    @functools.cached_property
    def initializer(self) -> mwpf.SolverInitializer:
        vertex_num = self._dem.num_detectors
        weighted_edges = [
            mwpf.HyperEdge(
                dem_hyperedge.detectors,
                probability_to_weight(dem_hyperedge.probability),
            )
            for dem_hyperedge in self.hyperedges
        ]
        return mwpf.SolverInitializer(vertex_num, weighted_edges)

    @functools.cached_property
    def predictor(self) -> "StaticPredictor":
        fault_masks = [
            sum(1 << k for k in dem_hyperedge.observables)
            for dem_hyperedge in self.hyperedges
        ]
        return StaticPredictor(
            fault_masks=fault_masks,
            num_dets=self._dem.num_detectors,
            num_obs=self._dem.num_observables,
        )


class Predictor(Protocol):
    """given the syndrome and the subgraph, predict the logical observable correction"""

    def syndrome_of(self, dets_bit_packed: np.ndarray) -> mwpf.SyndromePattern: ...
    def prediction_of(
        self, syndrome: mwpf.SyndromePattern, subgraph: Sequence[int]
    ) -> int: ...
    def num_detectors(self) -> int: ...
    def num_observables(self) -> int: ...

    def get_observable_bits(
        self, prediction: np.ndarray, shot_index: int = 0
    ) -> list[int]:
        return list(
            np.unpackbits(
                prediction[shot_index], count=self.num_observables(), bitorder="little"
            )
        )

    @property
    def is_dynamic(self) -> bool:
        return False


@dataclass(frozen=True)
class StaticPredictor(Predictor):
    fault_masks: list[int]
    num_dets: int
    num_obs: int

    def syndrome_of(self, dets_bit_packed: np.ndarray) -> mwpf.SyndromePattern:
        defect_vertices: list[int] = list(
            np.flatnonzero(
                np.unpackbits(dets_bit_packed, count=self.num_dets, bitorder="little")
            )
        )
        return mwpf.SyndromePattern(defect_vertices=defect_vertices)

    def prediction_of(
        self, _syndrome: mwpf.SyndromePattern, subgraph: Sequence[int]
    ) -> int:
        return reduce(lambda x, y: x ^ y, (self.fault_masks[i] for i in subgraph), 0)

    def num_detectors(self) -> int:
        return self.num_dets

    def num_observables(self) -> int:
        return self.num_obs


@dataclass(frozen=True)
class DemHyperedge:
    detectors: frozenset[int]
    observables: frozenset[int]
    probability: float


WEIGHT_MAX: float = 1000  # larger than np.log(1e300)


def probability_to_weight(probability: float) -> float:
    if probability >= 1:
        return -WEIGHT_MAX
    return np.log((1 - probability) / probability)


def weight_to_probability(weight: float) -> float:
    return 1 / (1 + np.exp(weight))


def exclusive_probability(p1: float, p2: float) -> float:
    return p1 * (1 - p2) + p2 * (1 - p1)


# tag is introduced in stim 1.15
HAS_TAG = hasattr(stim.CircuitInstruction, "tag")


def extra_instruction_params(instruction: Any) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if HAS_TAG:
        params["tag"] = instruction.tag
    return params
