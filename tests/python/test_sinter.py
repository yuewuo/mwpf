from common import *
import stim
import sinter
import numpy as np


params = dict(panic_action=mwpf.PanicAction.RAISE)


@pytest.mark.parametrize(
    "decoder",
    [
        mwpf.SinterMWPFDecoder(cluster_node_limit=50, **params),
        mwpf.SinterHUFDecoder(**params),
        mwpf.SinterMWPFDecoder(cluster_node_limit=0, **params),
    ],
)
@pytest.mark.parametrize(
    "p",
    [0.001],
)
@pytest.mark.parametrize(
    "d",
    [3],
)
@pytest.mark.parametrize(
    "code_type",
    [
        "color_code:memory_xyz",
        "repetition_code:memory",
        "surface_code:rotated_memory_x",
    ],
)
def test_sinter_decode(decoder: sinter.Decoder, p: float, d: int, code_type: str):
    if "mwpf" not in sys.modules:
        print("[skip] because sinter would fail to import without the mwpf package")
        print(
            "once https://github.com/quantumlib/Stim/pull/873 is merged, this test should be enabled"
        )
        return

    # pytest -s tests/python/test_sinter.py::test_sinter_decode
    circuit = stim.Circuit.generated(
        code_type,
        rounds=d,
        distance=d,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )

    task = sinter.Task(
        circuit=circuit,
        collection_options=sinter.CollectionOptions(max_shots=10, max_errors=10),
    )

    sinter.collect(
        num_workers=2,
        tasks=[task],
        decoders=["mwpf"],
        custom_decoders={"mwpf": decoder},
    )


def test_sinter_heralded_error():
    circuit = stim.Circuit(
        """\
R 0 1
X_ERROR(0.01) 0  # with lower probability, an error happens at 0
X_ERROR(0.1) 1  # normally we would always guess that the actual error happens at 1
HERALDED_PAULI_CHANNEL_1(0.02, 0.02, 0, 0) 0  # D0
DETECTOR rec[-1]
MPP Z0*Z1  # D1
DETECTOR rec[-1]
M 0
OBSERVABLE_INCLUDE(0) rec[-1]  # read qubit 0
"""
    )
    print(circuit)
    dem = circuit.detector_error_model(approximate_disjoint_errors=True)
    print("######### dem #########")
    print(dem)
    # without special handling, the dem includes the heralded error as regular pauli error
    assert dem == stim.DetectorErrorModel(
        """
error(0.02) D0
error(0.02) D0 D1 L0
error(0.1) D1
error(0.01) D1 L0
"""
    )
    decoder = mwpf.SinterMWPFDecoder(**params)
    compiled_decoder = decoder.compile_decoder_for_dem(dem=dem)
    # trigger both D0 and D1:
    bit_packed_detection_event_data = np.packbits(
        np.array([[1, 1]]), axis=-1, bitorder="little"
    )
    prediction = compiled_decoder.decode_shots_bit_packed(
        bit_packed_detection_event_data=bit_packed_detection_event_data
    )
    observables = compiled_decoder.predictor.get_observable_bits(prediction)
    assert observables == [1]

    decoder.with_circuit(circuit)
    compiled_decoder = decoder.compile_decoder_for_dem(dem=dem)
    prediction = compiled_decoder.decode_shots_bit_packed(
        bit_packed_detection_event_data=bit_packed_detection_event_data
    )
    observables = np.unpackbits(
        prediction[0], count=dem.num_observables, bitorder="little"
    )
    assert observables == [1]

    """
    In the above example, it seems like heralded errors are handled correctly in both cases.
    Let's see if we have construct cases where the regular fail but the normal one works.
    
    It's probably easier to not trigger the heralded detection. Normal handling of the error might
    treat the non-triggered event as "even parity check", and thus selecting two highly improbable
    cases
    """

    circuit2 = stim.Circuit(
        """\
R 0 1 2 3 4 5 6 7 8  # data qubits
MPP X0*X3*X6  # measure initial logical state
MPP X0*X1*X3*X4 X1*X2 X4*X5*X7*X8 X6*X7 # prepare stabilizer state
#  0  1  2
#  3  4  5
#  6  7  8

# add some noise
HERALDED_ERASE(0.3) 4  # the center qubits
DETECTOR rec[-1]
Z_ERROR(0.0001) 3 5  # these errors are less likely than the center erasure,
                   # however, if the herald indicator is False, we should still use this

# when the detector is not triggered, a regular decoder might think a Y error happens
MPP X0*X1*X3*X4 X1*X2 X4*X5*X7*X8 X6*X7
DETECTOR rec[-4] rec[-9]
DETECTOR rec[-3] rec[-8]
DETECTOR rec[-2] rec[-7]
DETECTOR rec[-1] rec[-6]
MPP Z0*Z3 Z1*Z2*Z4*Z5 Z3*Z4*Z6*Z7 Z5*Z8
DETECTOR rec[-4]
DETECTOR rec[-3]
DETECTOR rec[-2]
DETECTOR rec[-1]
MPP X0*X3*X6
OBSERVABLE_INCLUDE(0) rec[-1] rec[-15]
"""
    )
    print(circuit2)
    dem = circuit2.detector_error_model(approximate_disjoint_errors=True)
    print("######### dem #########")
    print(dem)
    assert dem == stim.DetectorErrorModel(
        """
error(0.075) D0
error(0.075) D0 D1 D3
error(0.075) D0 D1 D3 D6 D7
error(0.075) D0 D6 D7
error(0.0001) D1 L0
error(0.0001) D3
detector D2
detector D4
detector D5
detector D8
"""
    )
    decoder = mwpf.SinterMWPFDecoder(**params)
    compiled_decoder = decoder.compile_decoder_for_dem(dem=dem)
    # trigger D1 and D3
    bit_packed_detection_event_data = np.packbits(
        np.array([[0, 1, 0, 1, 0, 0, 0, 0, 0]]), axis=-1, bitorder="little"
    )
    prediction = compiled_decoder.decode_shots_bit_packed(
        bit_packed_detection_event_data=bit_packed_detection_event_data
    )
    observables = compiled_decoder.predictor.get_observable_bits(prediction)
    assert observables == [0]
    # on the other hand, a heralded-error-aware decoder should be able to figure out
    # that although the middle error is has higher probability, when the herald detector is False,
    # it doesn't really have any effect and thus we should choose the two error(0.0001) despite
    # that the probability of their both happening is pretty low
    decoder.with_circuit(circuit2)
    compiled_decoder = decoder.compile_decoder_for_dem(dem=dem)
    prediction = compiled_decoder.decode_shots_bit_packed(
        bit_packed_detection_event_data=bit_packed_detection_event_data
    )
    observables = np.unpackbits(
        prediction[0], count=dem.num_observables, bitorder="little"
    )
    assert observables == [1]
