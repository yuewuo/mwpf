from common import *
import stim


def test_ref_circuit_simple():
    circuit_str = """\
R 0 1 2 3 4
TICK
CX 0 1 2 3
TICK
CX 2 1 4 3
TICK
MR 1 3
DETECTOR(1, 0) rec[-2]
DETECTOR(3, 0) rec[-1]
HERALDED_ERASE(0.01) 0
DETECTOR rec[-1]
M 0 2 4
DETECTOR(1, 1) rec[-2] rec[-3] rec[-6]
DETECTOR(3, 1) rec[-1] rec[-2] rec[-5]
OBSERVABLE_INCLUDE(0) rec[-1]\
"""
    circuit = stim.Circuit(circuit_str)
    print("######### original circuit #########")
    print(circuit)
    assert str(circuit) == circuit_str
    ref_circuit = mwpf.ref_circuit.RefCircuit.of(circuit)
    print("######### absolute indexed circuit #########")
    print(ref_circuit)
    assert (
        str(ref_circuit)
        == """\
R 0 1 2 3 4
TICK
CX 0 1 2 3
TICK
CX 2 1 4 3
TICK
MR 1 3
DETECTOR(1, 0) abs[0]
DETECTOR(3, 0) abs[1]
HERALDED_ERASE(0.01) 0
DETECTOR abs[2]
M 0 2 4
DETECTOR(1, 1) abs[4] abs[3] abs[0]
DETECTOR(3, 1) abs[5] abs[4] abs[1]
OBSERVABLE_INCLUDE(0) abs[5]\
"""
    )
    circuit_2 = ref_circuit.circuit()
    print("######### new circuit #########")
    print(circuit_2)
    assert isinstance(circuit_2, stim.Circuit)
    assert str(circuit_2) == circuit_str

    # now let's play with some of the functionalities
    # first, the measurement recording array
    recs = ref_circuit.recs
    assert len(recs) == 6
    assert [rec.instruction.index(ref_circuit) for rec in recs] == [6, 6, 9, 11, 11, 11]
    assert [rec.abs_index(ref_circuit) for rec in recs] == [0, 1, 2, 3, 4, 5]
    assert recs[0].rel_index(ref_circuit, ref_circuit[7]) == -2
    assert recs[1].rel_index(ref_circuit, ref_circuit[7]) == -1
    # second, the detectors
    detectors = ref_circuit.detectors
    assert len(detectors) == 5
    assert [detector.num_measurements for detector in detectors] == [0] * 5
    assert [len(detector.targets) for detector in detectors] == [1, 1, 1, 3, 3]
    assert [detector.index(ref_circuit) for detector in detectors] == [7, 8, 10, 12, 13]
    assert [ref_circuit.detector_to_index[detector] for detector in detectors] == list(
        range(5)
    )
    assert ref_circuit.rec_to_detectors[recs[0]] == (detectors[0], detectors[3])
    assert ref_circuit.rec_to_detectors[recs[1]] == (detectors[1], detectors[4])
    assert ref_circuit.rec_to_detectors[recs[2]] == (detectors[2],)
    assert ref_circuit.rec_to_detectors[recs[3]] == (detectors[3],)
    assert ref_circuit.rec_to_detectors[recs[4]] == (detectors[3], detectors[4])
    assert ref_circuit.rec_to_detectors[recs[5]] == (detectors[4],)
    # then the instructions
    assert [
        ref_circuit.instruction_to_index[instruction] for instruction in ref_circuit
    ] == list(range(15))
    # finally play with the dem
    original_dem = circuit.detector_error_model(
        approximate_disjoint_errors=True, flatten_loops=True
    ).flattened()
    print("######### original dem #########")
    print(original_dem)
    print("######### new dem #########")
    new_dem = ref_circuit.ref_dem.dem()
    print(new_dem)
    print(new_dem == original_dem)


def test_ref_circuit_heralded_erase_example():
    circuit_str_with_comment = """\
HERALDED_ERASE(0.01) 0
# Declare a flag detector based on the erasure
DETECTOR rec[-1]

# Erase qubit 2 with 2% probability
# Separately, erase qubit 3 with 2% probability
HERALDED_ERASE(0.02) 2 3

# Do an XXXX measurement
MPP X2*X3*X5*X7
# Apply partially-heralded noise to the two qubits
HERALDED_ERASE(0.01) 2 3 5 7
DEPOLARIZE1(0.0001) 2 3 5 7
# Repeat the XXXX measurement
MPP X2*X3*X5*X7
# Declare a detector comparing the two XXXX measurements
DETECTOR rec[-1] rec[-6]
# Declare flag detectors based on the erasures
DETECTOR rec[-2]
DETECTOR rec[-3]
DETECTOR rec[-4]
DETECTOR rec[-5]\
"""
    circuit_str = """\
HERALDED_ERASE(0.01) 0
DETECTOR rec[-1]
HERALDED_ERASE(0.02) 2 3
MPP X2*X3*X5*X7
HERALDED_ERASE(0.01) 2 3 5 7
DEPOLARIZE1(0.0001) 2 3 5 7
MPP X2*X3*X5*X7
DETECTOR rec[-1] rec[-6]
DETECTOR rec[-2]
DETECTOR rec[-3]
DETECTOR rec[-4]
DETECTOR rec[-5]\
"""
    circuit = stim.Circuit(circuit_str_with_comment)
    print("######### original circuit #########")
    print(circuit)
    assert str(circuit) == circuit_str
    ref_circuit = mwpf.ref_circuit.RefCircuit.of(circuit)
    print("######### absolute indexed circuit #########")
    print(ref_circuit)
    assert (
        str(ref_circuit)
        == """\
HERALDED_ERASE(0.01) 0
DETECTOR abs[0]
HERALDED_ERASE(0.02) 2 3
MPP X2*X3*X5*X7
HERALDED_ERASE(0.01) 2 3 5 7
DEPOLARIZE1(0.0001) 2 3 5 7
MPP X2*X3*X5*X7
DETECTOR abs[8] abs[3]
DETECTOR abs[7]
DETECTOR abs[6]
DETECTOR abs[5]
DETECTOR abs[4]\
"""
    )
    circuit_2 = ref_circuit.circuit()
    print("######### new circuit #########")
    print(circuit_2)
    assert isinstance(circuit_2, stim.Circuit)
    assert str(circuit_2) == circuit_str


def test_ref_circuit_heralded_pauli_channel_1_example():
    circuit_str_with_comment = """\
# With 10% probability perform a phase flip of qubit 0.
HERALDED_PAULI_CHANNEL_1(0, 0, 0, 0.1) 0
DETECTOR rec[-1]  # Include the herald in detectors available to the decoder

# With 20% probability perform a heralded dephasing of qubit 0.
HERALDED_PAULI_CHANNEL_1(0.1, 0, 0, 0.1) 0
DETECTOR rec[-1]

# Subject a Bell Pair to heralded noise.
MXX 0 1
MZZ 0 1
HERALDED_PAULI_CHANNEL_1(0.01, 0.02, 0.03, 0.04) 0 1
MXX 0 1
MZZ 0 1
DETECTOR rec[-1] rec[-5]  # Did ZZ stabilizer change?
DETECTOR rec[-2] rec[-6]  # Did XX stabilizer change?
DETECTOR rec[-3]    # Did the herald on qubit 1 fire?
DETECTOR rec[-4]    # Did the herald on qubit 0 fire?\
"""
    circuit_str = """\
HERALDED_PAULI_CHANNEL_1(0, 0, 0, 0.1) 0
DETECTOR rec[-1]
HERALDED_PAULI_CHANNEL_1(0.1, 0, 0, 0.1) 0
DETECTOR rec[-1]
MXX 0 1
MZZ 0 1
HERALDED_PAULI_CHANNEL_1(0.01, 0.02, 0.03, 0.04) 0 1
MXX 0 1
MZZ 0 1
DETECTOR rec[-1] rec[-5]
DETECTOR rec[-2] rec[-6]
DETECTOR rec[-3]
DETECTOR rec[-4]\
"""
    circuit = stim.Circuit(circuit_str_with_comment)
    print("######### original circuit #########")
    print(circuit)
    assert str(circuit) == circuit_str
    ref_circuit = mwpf.ref_circuit.RefCircuit.of(circuit)
    print("######### absolute indexed circuit #########")
    print(ref_circuit)
    assert (
        str(ref_circuit)
        == """\
HERALDED_PAULI_CHANNEL_1(0, 0, 0, 0.1) 0
DETECTOR abs[0]
HERALDED_PAULI_CHANNEL_1(0.1, 0, 0, 0.1) 0
DETECTOR abs[1]
MXX 0 1
MZZ 0 1
HERALDED_PAULI_CHANNEL_1(0.01, 0.02, 0.03, 0.04) 0 1
MXX 0 1
MZZ 0 1
DETECTOR abs[7] abs[3]
DETECTOR abs[6] abs[2]
DETECTOR abs[5]
DETECTOR abs[4]\
"""
    )
    circuit_2 = ref_circuit.circuit()
    print("######### new circuit #########")
    print(circuit_2)
    assert isinstance(circuit_2, stim.Circuit)
    assert str(circuit_2) == circuit_str


def test_adding_circuit():
    circuit_1 = stim.Circuit("MXX 0 1")
    circuit_2 = stim.Circuit("MZZ 2 3")
    circuit = circuit_1 + circuit_2
    print(circuit)
    assert str(circuit) == "MXX 0 1\nMZZ 2 3"
    ref_circuit_1 = mwpf.ref_circuit.RefCircuit.of(circuit_1)
    ref_circuit_2 = mwpf.ref_circuit.RefCircuit.of(circuit_2)
    ref_circuit = ref_circuit_1 + ref_circuit_2
    print(ref_circuit)
    assert str(ref_circuit) == "MXX 0 1\nMZZ 2 3"

    # adding the same circuit should work but not the ref circuits
    circuit = circuit_1 + circuit_1
    print(circuit)
    assert str(circuit) == "MXX 0 1 0 1"
    try:
        ref_circuit = ref_circuit_1 + ref_circuit_1
    except BaseException:
        ...
    else:
        raise Exception("it should panic")
    ref_circuit_1_clone = ref_circuit_1.clone()
    ref_circuit = ref_circuit_1 + ref_circuit_1_clone
