import math
import pathlib
from typing import Tuple, Any, Optional
import mwpf
from mwpf import (  # type: ignore
    SyndromePattern,
    HyperEdge,
    SolverInitializer,
    Solver,
    BP,
    BenchmarkSuite,
    WeightRange,
)
from dataclasses import dataclass, field
import pickle
import json
import traceback
from enum import Enum
import random
import numpy as np
import stim
from io import BufferedReader, BufferedWriter
import time
import struct
from contextlib import nullcontext
from .ref_circuit import *
from .heralded_dem import *

available_decoders = [
    "Solver",  # the solver with the highest accuracy, but may change across different versions
    "SolverSerialJointSingleHair",
    "SolverSerialSingleHair",
    "SolverSerialUnionFind",
]

default_cluster_node_limit: int = 50


@dataclass
class DecoderPanic:
    initializer: SolverInitializer
    config: dict
    syndrome: SyndromePattern
    panic_message: str


class PanicAction(Enum):
    RAISE = 1  # raise the panic with proper message to help debugging
    CATCH = 2  # proceed with normal decoding and return all-0 result


@dataclass
class SinterMWPFDecoder:
    """
    Use MWPF to predict observables from detection events.

    Args:
        decoder_type: decoder class used to construct the MWPF decoder.  in the Rust implementation, all of them inherits from the class of `SolverSerialPlugins` but just provide different plugins for optimizing the primal and/or dual solutions. For example, `SolverSerialUnionFind` is the most basic solver without any plugin: it only grows the clusters until the first valid solution appears; some more optimized solvers uses one or more plugins to further optimize the solution, which requires longer decoding time.

        cluster_node_limit (alias: c): The maximum number of nodes in a cluster, used to tune the performance of the decoder. The default value is 50.
    """

    decoder_type: str = "SolverSerialJointSingleHair"
    cluster_node_limit: Optional[int] = None
    c: Optional[int] = None  # alias of `cluster_node_limit`, will override it
    timeout: Optional[float] = None
    with_progress: bool = False
    circuit: Optional[stim.Circuit] = None  # RefCircuit is not picklable
    # this parameter itself doesn't do anything to load the circuit but only check whether the circuit is indeed loaded
    pass_circuit: bool = False

    # record panic data and controls whether the raise the panic or simply record them
    panic_action: PanicAction = PanicAction.CATCH
    panic_cases: list[DecoderPanic] = field(default_factory=list)

    # record benchmark suite when enabled
    benchmark_suite_filename: Optional[str] = None
    # record decoding data when enabled, including decoding time, primal and dual weights, etc.
    trace_filename: Optional[str] = None

    # adding BP decoder as a pre-decoder
    bp: bool = False
    max_iter: int = 0  # by default "adaptive"
    bp_method: str = "ms"  # 'product_sum' or 'minimum_sum'
    ms_scaling_factor: float = 0.625  # usually better than the original default of 1.0
    schedule: str = "parallel"  #  'parallel', 'serial', or 'serial_relative'
    omp_thread_count: int = 1
    random_schedule_seed: int = 0
    serial_schedule_order: Optional[list[int]] = None
    bp_weight_mix_ratio: float = 1.0
    floor_weight: Optional[float] = (
        None  # when updating the mwpf weights, all the weights are enforced to be no less than this value; by default 0 which enforces all the weights to be non-negative
    )
    # by default BP may converge and directly return the result;
    # sometimes it's better if BP cannot directly return and always use the post-processing to decode
    bp_converge: bool = True

    @property
    def _cluster_node_limit(self) -> int:
        if self.cluster_node_limit is not None:
            assert self.c is None, "Cannot set both `cluster_node_limit` and `c`."
            return self.cluster_node_limit
        elif self.c is not None:
            assert (
                self.cluster_node_limit is None
            ), "Cannot set both `cluster_node_limit` and `c`."
            return self.c
        return default_cluster_node_limit

    @property
    def config(self) -> dict[str, Any]:
        return dict(cluster_node_limit=self._cluster_node_limit)

    def with_circuit(self, circuit: stim.Circuit | None) -> "SinterMWPFDecoder":
        if circuit is None:
            self.circuit = None
            return self
        assert isinstance(circuit, stim.Circuit)
        self.circuit = circuit.copy()
        return self

    def common_prepare(
        self, dem: "stim.DetectorErrorModel"
    ) -> tuple[Any, Predictor, Any]:
        if self.pass_circuit:
            assert (
                self.circuit is not None
            ), "The circuit is not loaded but the flag `pass_circuit` is True"

        solver, predictor = construct_decoder_and_predictor(
            dem,
            decoder_type=self.decoder_type,
            config=self.config,
            ref_circuit=(
                RefCircuit.of(self.circuit) if self.circuit is not None else None
            ),
        )
        assert (
            dem.num_detectors == predictor.num_detectors()
        ), "Mismatched number of detectors, are you using the corresponding circuit of dem?"
        assert (
            dem.num_observables == predictor.num_observables()
        ), "Mismatched number of observables, are you using the corresponding circuit of dem?"

        # construct bp decoder if requested
        bp_decoder: Optional[Any] = None
        if self.bp:
            if self.circuit is not None:
                assert (
                    not predictor.is_dynamic
                ), "BP is not supported for dynamic predictors, e.g., in presence of heralded errors."

            from ldpc import BpDecoder
            from ldpc.ckt_noise.dem_matrices import (
                detector_error_model_to_check_matrices,
            )

            bp_matrices = detector_error_model_to_check_matrices(
                dem, allow_undecomposed_hyperedges=True
            )
            bp_decoder = BpDecoder(
                pcm=bp_matrices.check_matrix,
                error_channel=list(bp_matrices.priors),
                max_iter=self.max_iter,
                bp_method=self.bp_method,
                ms_scaling_factor=self.ms_scaling_factor,
                schedule=self.schedule,
                omp_thread_count=self.omp_thread_count,
                serial_schedule_order=self.serial_schedule_order,
                input_vector_type="syndrome",
            )

        return solver, predictor, bp_decoder

    def compile_decoder_for_dem(
        self,
        *,
        dem: "stim.DetectorErrorModel",
    ) -> "MwpfCompiledDecoder":
        solver, predictor, bp_decoder = self.common_prepare(dem)

        benchmark_suite: Optional[BenchmarkSuite] = None
        if self.benchmark_suite_filename is not None:
            benchmark_suite = BenchmarkSuite(solver.get_initializer())
            solver = None

        return MwpfCompiledDecoder(
            solver,
            predictor,
            dem.num_detectors,
            dem.num_observables,
            panic_action=self.panic_action,
            panic_cases=self.panic_cases,  # record all the panic information to the same place
            benchmark_suite=benchmark_suite,
            benchmark_suite_filename=self.benchmark_suite_filename,
            trace_filename=self.trace_filename,
            bp_decoder=bp_decoder,
            bp_weight_mix_ratio=self.bp_weight_mix_ratio,
            floor_weight=self.floor_weight,
            bp_converge=self.bp_converge,
        )

    def decode_via_files(
        self,
        *,
        num_shots: int,
        num_dets: int,
        num_obs: int,
        dem_path: pathlib.Path,
        dets_b8_in_path: pathlib.Path,
        obs_predictions_b8_out_path: pathlib.Path,
        tmp_dir: pathlib.Path,
    ) -> None:
        dem = stim.DetectorErrorModel.from_file(dem_path)

        solver, predictor, bp_decoder = self.common_prepare(dem)

        assert num_dets == predictor.num_detectors()
        assert num_obs == predictor.num_observables()

        benchmark_suite: Optional[BenchmarkSuite] = None
        if self.benchmark_suite_filename is not None:
            benchmark_suite = BenchmarkSuite(solver.get_initializer())
            solver = None

        num_det_bytes = math.ceil(num_dets / 8)
        with (
            open(self.trace_filename, "wb")
            if self.trace_filename is not None
            else nullcontext()
        ) as trace_f:
            with open(dets_b8_in_path, "rb") as dets_in_f:
                with open(obs_predictions_b8_out_path, "wb") as obs_out_f:
                    for dets_bit_packed in iter_det(
                        dets_in_f, num_shots, num_det_bytes, self.with_progress
                    ):
                        prediction = decode_common(
                            dets_bit_packed=dets_bit_packed,
                            predictor=predictor,
                            solver=solver,
                            num_dets=num_dets,
                            num_obs=num_obs,
                            panic_action=self.panic_action,
                            panic_cases=self.panic_cases,
                            benchmark_suite=benchmark_suite,
                            trace_f=trace_f,
                            bp_decoder=bp_decoder,
                            bp_weight_mix_ratio=self.bp_weight_mix_ratio,
                            floor_weight=self.floor_weight,
                            bp_converge=self.bp_converge,
                        )
                        obs_out_f.write(
                            int(prediction).to_bytes(
                                (num_obs + 7) // 8, byteorder="little"
                            )
                        )

        if benchmark_suite is not None:
            benchmark_suite.save_cbor(self.benchmark_suite_filename)


def iter_det(
    f: BufferedReader,
    num_shots: int,
    num_det_bytes: int,
    with_progress: bool = False,
) -> Iterable[np.ndarray]:
    if with_progress:
        from tqdm import tqdm

        pbar = tqdm(total=num_shots, desc="shots")
    for _ in range(num_shots):
        if with_progress:
            pbar.update(1)
        dets_bit_packed = np.fromfile(f, dtype=np.uint8, count=num_det_bytes)
        if dets_bit_packed.shape != (num_det_bytes,):
            raise IOError("Missing dets data.")
        yield dets_bit_packed


def construct_decoder_and_predictor(
    model: "stim.DetectorErrorModel",
    decoder_type: Any,
    config: dict[str, Any],
    ref_circuit: Optional[RefCircuit] = None,
) -> Tuple[Any, Predictor]:

    if ref_circuit is not None:
        heralded_dem = HeraldedDetectorErrorModel(ref_circuit=ref_circuit)
        initializer = heralded_dem.initializer
        predictor: Predictor = heralded_dem.predictor
    else:
        ref_dem = RefDetectorErrorModel.of(dem=model)
        initializer = ref_dem.initializer
        predictor = ref_dem.predictor

    if decoder_type is None:
        # default to the solver with highest accuracy
        decoder_cls = Solver
    elif isinstance(decoder_type, str):
        decoder_cls = getattr(mwpf, decoder_type)
    else:
        decoder_cls = decoder_cls
    return (
        decoder_cls(initializer, config=config),
        predictor,
    )


def panic_text_of(solver, syndrome) -> str:
    initializer = solver.get_initializer()
    config = solver.config
    syndrome
    panic_text = f"""
######## MWPF Sinter Decoder Panic ######## 
solver_initializer: dict = json.loads('{initializer.to_json()}')
config: dict = json.loads('{json.dumps(config)}')
syndrome: dict = json.loads('{syndrome.to_json()}')
######## PICKLE DATA ######## 
solver_initializer: SolverInitializer = pickle.loads({pickle.dumps(initializer)!r})
config: dict = pickle.loads({pickle.dumps(config)!r})
syndrome: SyndromePattern = pickle.loads({pickle.dumps(syndrome)!r})
######## End Panic Information ######## 
"""
    return panic_text


@dataclass
class SinterHUFDecoder(SinterMWPFDecoder):
    decoder_type: str = "SolverSerialUnionFind"
    cluster_node_limit: int = 0


@dataclass
class SinterSingleHairDecoder(SinterMWPFDecoder):
    decoder_type: str = "SolverSerialSingleHair"
    cluster_node_limit: int = 0


@dataclass
class MwpfCompiledDecoder:
    solver: Any
    predictor: Predictor
    num_dets: int
    num_obs: int
    panic_action: PanicAction
    panic_cases: list[DecoderPanic]
    benchmark_suite: Optional[BenchmarkSuite]
    trace_filename: Optional[str]
    benchmark_suite_filename: Optional[str]
    bp_decoder: Any
    bp_weight_mix_ratio: float
    floor_weight: Optional[float]
    bp_converge: bool = True

    def decode_shots_bit_packed(
        self,
        *,
        bit_packed_detection_event_data: "np.ndarray",
    ) -> "np.ndarray":
        num_shots = bit_packed_detection_event_data.shape[0]
        predictions = np.zeros(
            shape=(num_shots, (self.num_obs + 7) // 8), dtype=np.uint8
        )
        with (
            open(self.trace_filename, "wb")
            if self.trace_filename is not None
            else nullcontext()
        ) as trace_f:
            for shot in range(num_shots):
                dets_bit_packed = bit_packed_detection_event_data[shot]
                prediction = decode_common(
                    dets_bit_packed=dets_bit_packed,
                    predictor=self.predictor,
                    solver=self.solver,
                    num_dets=self.num_dets,
                    num_obs=self.num_obs,
                    panic_action=self.panic_action,
                    panic_cases=self.panic_cases,
                    benchmark_suite=self.benchmark_suite,
                    trace_f=trace_f,
                    bp_decoder=self.bp_decoder,
                    bp_weight_mix_ratio=self.bp_weight_mix_ratio,
                    floor_weight=self.floor_weight,
                    bp_converge=self.bp_converge,
                )
                predictions[shot] = np.packbits(
                    np.array(
                        list(np.binary_repr(prediction, width=self.num_obs))[::-1],
                        dtype=np.uint8,
                    ),
                    bitorder="little",
                )

        if self.benchmark_suite is not None:
            self.benchmark_suite.save_cbor(self.benchmark_suite_filename)

        return predictions


def decode_common(
    dets_bit_packed: np.ndarray,
    predictor: Predictor,
    solver: Any,
    num_dets: int,
    num_obs: int,
    panic_action: PanicAction,
    panic_cases: list[DecoderPanic],
    benchmark_suite: Optional[BenchmarkSuite],
    trace_f: Optional[BufferedWriter],
    bp_decoder: Any,
    bp_weight_mix_ratio: float,
    floor_weight: Optional[float],
    bp_converge: bool = True,
) -> int:
    syndrome = predictor.syndrome_of(dets_bit_packed)
    if solver is None:
        if benchmark_suite is not None:
            benchmark_suite.append(syndrome)
        prediction = 0
    else:
        try:
            start = time.perf_counter()
            if bp_decoder is not None:
                dets_bits = np.unpackbits(
                    dets_bit_packed, count=num_dets, bitorder="little"
                )
                bp_solution = bp_decoder.decode(dets_bits)
                if bp_decoder.converge and bp_converge:
                    prediction = predictor.prediction_of(
                        syndrome, np.flatnonzero(bp_solution)
                    )
                else:
                    syndrome = SyndromePattern(
                        defect_vertices=syndrome.defect_vertices,
                        override_weights=list(bp_decoder.log_prob_ratios),
                        override_ratio=bp_weight_mix_ratio,
                        floor_weight=floor_weight,
                    )
                    solver.solve(syndrome)
                    if trace_f is None:
                        subgraph = solver.subgraph()
                    else:
                        subgraph, bound = solver.subgraph_range()
                        record_trace(trace_f, time.perf_counter() - start, bound)
                    prediction = predictor.prediction_of(syndrome, subgraph)
            else:
                solver.solve(syndrome)
                if trace_f is None:
                    subgraph = solver.subgraph()
                else:
                    subgraph, bound = solver.subgraph_range()
                    record_trace(trace_f, time.perf_counter() - start, bound)
                prediction = predictor.prediction_of(syndrome, subgraph)
        except BaseException as e:
            panic_cases.append(
                DecoderPanic(
                    initializer=solver.get_initializer(),
                    config=solver.config,
                    syndrome=syndrome,
                    panic_message=traceback.format_exc(),
                )
            )
            if "<class 'KeyboardInterrupt'>" in str(e):
                raise e
            elif panic_action == PanicAction.RAISE:
                raise e
                raise ValueError(panic_text_of(self.solver, syndrome)) from e
            elif panic_action == PanicAction.CATCH:
                prediction = random.getrandbits(num_obs)
    return prediction


def record_trace(trace_f: BufferedWriter, elapsed: float, bound: WeightRange):
    trace_f.write(struct.pack("f", elapsed))
    trace_f.write(struct.pack("f", bound.lower.float()))
    trace_f.write(struct.pack("f", bound.upper.float()))
    trace_f.write(struct.pack("f", 0))  # reserved
