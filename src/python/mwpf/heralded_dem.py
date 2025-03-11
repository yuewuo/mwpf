"""
Regular Detector Error Model (DEM) does not contain the information of the heralded error.
This HeraldedDetectorErrorModel class provides additional information on the heralded errors.
It is capable of reading bits from the detector which corresponds to the heralded error indicator.

Note that in order to let the tool read a heralded error in the circuit, it is required that 
the heralded error is detected using `DETECTOR rec[...]` where `rec[...]` corresponds to the heralded event.
To help user, we provide a function that automatically adds such detections.

[warning] not all circuits can be used in this class. Notably, there are several requirements:
1. the measurement of a heralded error must be either not detected or uniquely detected by a detector
``
"""

import stim
from .ref_circuit import (
    RefCircuit,
    RefInstruction,
    RefRec,
    RefDetector,
    RefDetectorErrorModel,
    probability_to_weight,
    Predictor,
)
from typing import Sequence
import functools
from dataclasses import dataclass
from frozendict import frozendict
import mwpf
import numpy as np


DEM_MIN_PROBABILITY = 1e-15  # below this value, DEM starts to ignore the error rate


# avoid non-zero small probability to be ignored by the DEM
def dem_probability(probability: float) -> float:
    if probability == 0:
        return 0.0
    return max(DEM_MIN_PROBABILITY, probability)


@dataclass(frozen=True)
class HeraldedDetectorErrorModel:
    ref_circuit: RefCircuit
    # We assume that the detector error model has a non-zero false positive rate to make the decoding
    # graph generation simpler. By default the value is 1e-300
    false_positive_rate: float = DEM_MIN_PROBABILITY

    def __post_init__(self) -> None:
        self.sanity_check()

    def of(
        circuit: stim.Circuit,
        false_positive_rate: float = DEM_MIN_PROBABILITY,
    ) -> "HeraldedDetectorErrorModel":
        return HeraldedDetectorErrorModel(
            ref_circuit=RefCircuit.of(circuit), false_positive_rate=false_positive_rate
        )

    def sanity_check(self) -> None:
        # check basic type
        assert isinstance(
            self.ref_circuit, RefCircuit
        ), "ref_circuit must be a RefCircuit"
        assert isinstance(
            self.false_positive_rate, float
        ), "false_positive_rate must be a float"
        # each heralded error is either not detected or uniquely detected
        for heralded_rec in self.heralded_measurements:
            assert (
                len(self.ref_circuit.rec_to_detectors[heralded_rec]) <= 1
            ), f"abs[{heralded_rec.abs_index(self.ref_circuit)}] is detected by multiple detectors"
            for detector in self.ref_circuit.rec_to_detectors[heralded_rec]:
                assert len(detector.targets) == 1, (
                    f"detector[{self.ref_circuit.detector_to_index[detector]}] detects multiple recs; "
                    + "we require that the detector of a heralded error must only detect one rec"
                )

    @functools.cached_property
    def heralded_instructions(self) -> tuple[RefInstruction, ...]:
        return tuple(
            instruction
            for instruction in self.ref_circuit
            if is_heralded_error(instruction)
        )

    @functools.cached_property
    def heralded_measurements(self) -> tuple[RefRec, ...]:
        return tuple(
            rec
            for instruction in self.ref_circuit
            if is_heralded_error(instruction)
            for rec in instruction.recs
        )

    @functools.cached_property
    def detected_heralded_measurements(self) -> tuple[RefRec, ...]:
        return tuple(
            rec
            for rec in self.heralded_measurements
            if self.ref_circuit.rec_to_detectors[rec]
        )

    @functools.cached_property
    def undetected_heralded_measurements(self) -> tuple[RefRec, ...]:
        return tuple(
            rec
            for rec in self.heralded_measurements
            if not self.ref_circuit.rec_to_detectors[rec]
        )

    @functools.cached_property
    def heralded_detectors(self) -> tuple[RefDetector | None, ...]:
        heralded_measurements = frozenset(self.heralded_measurements)
        return tuple(
            detector if (set(detector.targets) & heralded_measurements) else None
            for detector in self.ref_circuit.detectors
        )

    @functools.cached_property
    def heralded_detector_indices(self) -> tuple[int, ...]:
        return tuple(
            {
                detector_id
                for detector_id, detector in enumerate(self.heralded_detectors)
                if detector is not None
            }
        )

    @functools.cached_property
    def detector_id_to_herald_id(self) -> frozendict[int, int]:
        return frozendict(
            {
                detector_id: herald_id
                for herald_id, detector_id in enumerate(self.heralded_detector_indices)
            }
        )

    @functools.cached_property
    def num_heralds(self) -> int:
        return len(self.heralded_detector_indices)

    @functools.cached_property
    def skeleton_circuit(self) -> RefCircuit:
        """
        The skeleton circuit is a circuit where all the heralded errors are not triggered.
        There will be still some false positive rate remaining to make sure all the possible
        hyperedges still exist in the decoding hypergraph.
        """
        new_instructions = list(self.ref_circuit)
        deleting_indices: list[int] = []
        # first change instruction:
        #     HERALDED_ERASE -> DEPOLARIZE1(false_positive_rate)
        #     HERALDED_PAULI_CHANNEL_1 -> PAULI_CHANNEL_1(...)
        for instruction in self.heralded_instructions:
            instruction_index = self.ref_circuit.instruction_to_index[instruction]
            noise_instruction = heralded_instruction_to_noise_instruction(instruction)
            if noise_instruction is None:
                deleting_indices.append(instruction_index)
                continue
            tiny_noise_instruction = RefInstruction(
                name=noise_instruction.name,
                targets=noise_instruction.targets,
                gate_args=tuple(
                    dem_probability(p * self.false_positive_rate)
                    for p in noise_instruction.gate_args
                ),
            )
            new_instructions[instruction_index] = tiny_noise_instruction
        # then delete detectors of the heralded errors
        for detector in self.heralded_detectors:
            if detector is not None:
                deleting_indices.append(self.ref_circuit.instruction_to_index[detector])
        assert len(set(deleting_indices)) == len(deleting_indices), "bug: duplicate"
        for index in sorted(deleting_indices, reverse=True):
            del new_instructions[index]
        return RefCircuit.of(new_instructions)

    @functools.cached_property
    def skeleton_dem(self) -> RefDetectorErrorModel:
        """
        construct a dem whose detector id corresponds to the detectors of the original circuit
        instead of the skeleton circuit
        """
        ref_dem = self.skeleton_circuit.ref_dem
        # let the dem refer to self.ref_circuit instead
        return RefDetectorErrorModel(
            instructions=ref_dem.instructions, ref_circuit=self.ref_circuit
        )

    @functools.cached_property
    def heralded_dems(
        self,
    ) -> frozendict[RefDetector, RefDetectorErrorModel]:
        ref_dems: dict[RefDetector, RefDetectorErrorModel] = {}
        for detector in self.heralded_detectors:
            if detector is None:
                continue
            ref_rec = detector.targets[0]
            heralded_instruction = ref_rec.instruction
            all_noise_instruction = heralded_instruction_to_noise_instruction(
                heralded_instruction
            )
            if all_noise_instruction is None:
                continue
            assert isinstance(ref_rec, RefRec)
            # remove all the noise channels except the heralded error
            circuit_no_noise = self.ref_circuit.remove_noise_channels(
                keeping={ref_rec.instruction}
            )
            new_circuit_instructions = list(circuit_no_noise)
            # change the heralded error to the noise channel when it happens
            assert (
                len(all_noise_instruction.targets)
                == heralded_instruction.num_measurements
            ), "the following code assumes target has a heralding measurement"
            new_circuit_instructions[
                circuit_no_noise.instruction_to_index[heralded_instruction]
            ] = RefInstruction(
                name=all_noise_instruction.name,
                targets=(all_noise_instruction.targets[ref_rec.bias],),
                gate_args=all_noise_instruction.gate_args,
            )
            new_circuit = RefCircuit.of(new_circuit_instructions)
            heralded_dem = RefDetectorErrorModel(
                instructions=new_circuit.ref_dem.instructions,
                ref_circuit=self.ref_circuit,
            )
            if not heralded_dem.hyperedges:
                # if there is no hyperedge, we don't need this detector at all
                continue
            for hyperedge in heralded_dem.hyperedges:
                assert (
                    hyperedge.detectors in self.skeleton_dem.hyperedges_detectors_set
                ), (
                    "bug: the skeleton graph doesn't have the hyperedge, "
                    + "this might causes issue when constructing decoders"
                )
            ref_dems[detector] = heralded_dem

        return frozendict(ref_dems)

    def __str__(self) -> str:
        result = "HeraldedDetectorErrorModel:"
        result += "\n    skeleton hypergraph:"
        for dem_hyperedge in self.skeleton_dem.hyperedges:
            result += (
                f"\n        {', '.join([f'D{v}' for v in sorted(dem_hyperedge.detectors)])}: {dem_hyperedge.probability:.6e}"
                + f" ({', '.join([f'L{v}' for v in sorted(dem_hyperedge.observables)])})"
            )
        for detector, ref_dem in self.heralded_dems.items():
            result += f"\n    heralded hypergraph on D{self.ref_circuit.detector_to_index[detector]}:"
            for hyperedge in ref_dem.hyperedges:
                result += (
                    f"\n        {', '.join([f'D{v}' for v in sorted(hyperedge.detectors)])}: {hyperedge.probability:.6e}"
                    + f" ({', '.join([f'L{v}' for v in sorted(hyperedge.observables)])})"
                )
        return result

    @functools.cached_property
    def hyperedge_to_index(self) -> frozendict[frozenset[int], int]:
        return frozendict(
            {
                dem_hyperedge.detectors: edge_index
                for edge_index, dem_hyperedge in enumerate(self.skeleton_dem.hyperedges)
            }
        )

    @functools.cached_property
    def herald_fault_map(self) -> tuple[frozendict[int, tuple[float, int]], ...]:
        heralds: list[frozendict[int, tuple[float, int]]] = []
        for detector_id in self.heralded_detector_indices:
            detector = self.heralded_detectors[detector_id]
            assert detector is not None
            if detector not in self.heralded_dems:
                heralds.append(frozendict({}))
                continue
            sub_dem = self.heralded_dems[detector]
            heralds.append(
                frozendict(
                    {
                        self.hyperedge_to_index[hyperedge.detectors]: (
                            hyperedge.probability,
                            sum(1 << k for k in hyperedge.observables),
                        )
                        for hyperedge in sub_dem.hyperedges
                    }
                )
            )
        return tuple(heralds)

    @functools.cached_property
    def initializer(self) -> mwpf.SolverInitializer:
        vertex_num = self.skeleton_dem._dem.num_detectors
        weighted_edges = [
            mwpf.HyperEdge(
                dem_hyperedge.detectors,
                probability_to_weight(dem_hyperedge.probability),
            )
            for dem_hyperedge in self.skeleton_dem.hyperedges
        ]
        heralds = [
            {edge_index: probability_to_weight(p) for edge_index, (p, _) in dic.items()}
            for dic in self.herald_fault_map
        ]
        return mwpf.SolverInitializer(vertex_num, weighted_edges, heralds=heralds)

    @functools.cached_property
    def predictor(self) -> "HeraldedDemPredictor":
        fault_masks_with_p = tuple(
            (sum(1 << k for k in dem_hyperedge.observables), dem_hyperedge.probability)
            for dem_hyperedge in self.skeleton_dem.hyperedges
        )
        herald_detectors = frozenset(
            {
                detector_id
                for detector_id, detector in enumerate(self.heralded_detectors)
                if detector is not None
            }
        )
        return HeraldedDemPredictor(
            fault_masks_with_p=fault_masks_with_p,
            herald_detectors=herald_detectors,
            detector_id_to_herald_id=self.detector_id_to_herald_id,
            herald_fault_map=self.herald_fault_map,
            num_dets=self.skeleton_dem._dem.num_detectors,
            num_obs=self.skeleton_dem._dem.num_observables,
        )


@dataclass(frozen=True)
class HeraldedDemPredictor(Predictor):
    """
    the correction should be chosen based on the heralded error: if certain observable achieves higher
    probability, we should choose the correction based on that observable; this is a dynamic behavior
    ideally, we should not spend too much computation in Python.
    How about given the subgraph object and then calculate the prediction? The subgraph should be pretty sparse
    """

    fault_masks_with_p: tuple[tuple[int, float], ...]
    herald_detectors: frozenset[int]
    detector_id_to_herald_id: frozendict[int, int]
    herald_fault_map: tuple[frozendict[int, tuple[float, int]], ...]
    num_dets: int
    num_obs: int

    def syndrome_of(self, dets_bit_packed: np.ndarray) -> mwpf.SyndromePattern:
        detectors: set[int] = set(
            np.flatnonzero(
                np.unpackbits(dets_bit_packed, count=self.num_dets, bitorder="little")
            )
        )
        # the heralded detectors are not passed to the decoder
        defect_vertices = detectors - self.herald_detectors
        # instead, they are passed as heralds
        heralds = [
            self.detector_id_to_herald_id[detector_id]
            for detector_id in detectors & self.herald_detectors
        ]
        return mwpf.SyndromePattern(defect_vertices=defect_vertices, heralds=heralds)

    def prediction_of(
        self, syndrome: mwpf.SyndromePattern, subgraph: Sequence[int]
    ) -> int:
        prediction: int = 0
        heralds: list[int] = syndrome.heralds
        for edge_index in subgraph:
            fault_mask, p = self.fault_masks_with_p[edge_index]
            # also iterate over the heralded errors to see if there is a more probable logical correction
            for herald_id in heralds:
                if edge_index in self.herald_fault_map[herald_id]:
                    new_p, new_fault_mask = self.herald_fault_map[herald_id][edge_index]
                    if new_p > p:
                        p = new_p
                        fault_mask = new_fault_mask
            prediction ^= fault_mask
        return prediction

    def num_detectors(self) -> int:
        return self.num_dets

    def num_observables(self) -> int:
        return self.num_obs

    @property
    def is_dynamic(self) -> bool:
        return len(self.herald_detectors) > 0


def add_herald_detectors(circuit: stim.Circuit) -> stim.Circuit:
    """
    Add detectors for heralded errors to the circuit, if they do not present in the original circuit.
    Note that the circuit will be fully expanded after this function. The heralded detectors added
    will be following each heralded error instruction.
    """
    ref_circuit = RefCircuit.of(circuit)
    heralded_instructions = tuple(
        instruction for instruction in ref_circuit if is_heralded_error(instruction)
    )
    new_instructions = list(ref_circuit)
    for heralded_instruction in reversed(heralded_instructions):
        for rec in reversed(heralded_instruction.recs):
            if not ref_circuit.rec_to_detectors[rec]:
                new_instructions.insert(
                    ref_circuit.instruction_to_index[heralded_instruction] + 1,
                    RefInstruction(
                        name="DETECTOR",
                        targets=(rec,),
                    ),
                )
    return RefCircuit.of(new_instructions).circuit()


def remove_herald_detectors(circuit: stim.Circuit) -> stim.Circuit:
    """
    Remove all detectors of the heralded errors. It will only remove detectors that uniquely detects the heralded errors,
    and panic if some composite detectors involve the heralded error measurement.
    """
    ref_circuit = RefCircuit.of(circuit)
    new_instructions = []
    for instruction in ref_circuit:
        if instruction.name != "DETECTOR":
            new_instructions.append(instruction)
            continue
        assert len(instruction.targets) > 0, "bug: detector without target"
        if len(instruction.targets) > 1:
            # check that none of these recs corresponds to an heralded instruction
            for rec in instruction.targets:
                assert isinstance(rec, RefRec)
                assert not is_heralded_error(rec.instruction), ""
            new_instructions.append(instruction)
            continue
        rec = instruction.targets[0]
        assert isinstance(rec, RefRec)
        if not is_heralded_error(rec.instruction):
            new_instructions.append(instruction)
    return RefCircuit.of(new_instructions).circuit()


def is_heralded_error(instruction: RefInstruction) -> bool:
    if instruction.name in ["HERALDED_ERASE", "HERALDED_PAULI_CHANNEL_1"]:
        return True
    if "HERALDED" in instruction.name:
        raise NotImplementedError(
            f"Instruction {instruction.name} has 'HERALDED' in its name but is not implemented yet."
        )
    return False


def heralded_instruction_to_noise_instruction(
    instruction: RefInstruction,
) -> RefInstruction | None:
    if instruction.name == "HERALDED_ERASE":
        heralded_probability = instruction.gate_args[0]
        if heralded_probability == 0:
            return None
        return RefInstruction(
            name="DEPOLARIZE1",
            targets=instruction.targets,
            gate_args=(0.75,),
        )
    elif instruction.name == "HERALDED_PAULI_CHANNEL_1":
        pI, pX, pY, pZ = instruction.gate_args
        if pX + pY + pZ == 0:
            return None
        p_sum = pI + pX + pY + pZ
        return RefInstruction(
            name="PAULI_CHANNEL_1",
            targets=instruction.targets,
            gate_args=(pX / p_sum, pY / p_sum, pZ / p_sum),
        )
    return None
