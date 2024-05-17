import itertools
from qulacs import QuantumState, QuantumCircuit
from qulacs.gate import Measurement, H, CNOT, CZ, X
from tool.testing import run_test
import os

def print_state(state: QuantumState, computational_states: list[str], num_data: int = 7) -> None:
    """
    Print the quantum state vector with corresponding computational states.

    Args:
        state (QuantumState): The quantum state vector.
        computational_states (list[str]): List of computational states.
        num_data (int): Number of data qubits.

    Returns:
        None
    """
    for i, s in enumerate(computational_states):
        s = s[:num_data] + '|' + s[num_data:]
        if abs(state[i]) > 1e-6:
            print(f'{s}: {state[i].real:.3f}  +  {state[i].imag:.3f}i')

gatedict = {
    'H': H,
    'CX': CNOT,
    'CZ': CZ,
    'Meas': Measurement,
}

def run_stabilizer_circuit(gate_sequence: list[tuple[str, tuple[int]]], num_qubits: int, num_meas: int,
                           target_outcomes: str = None, verbose: bool = False) -> tuple[QuantumState, str]:
    """
    Run a stabilizer circuit with the given gate sequence.

    Args:
        gate_sequence (list[tuple[str, tuple[int]]]): List of gates and their positions.
        num_qubits (int): Number of qubits in the circuit.
        num_meas (int): Number of measurement qubits.
        target_outcomes (str): Target outcomes for the ancilla measurements.
        verbose (bool): Whether to print verbose output.

    Returns:
        tuple[QuantumState, str]: The final quantum state and the ancilla measurement outcomes.
    """
    computational_states = [''.join(map(str,i))[::-1] for i in itertools.product([0, 1], repeat=num_qubits)]
    anc_meas = ''

    while anc_meas != target_outcomes:

        state = QuantumState(num_qubits)
        state.set_zero_state()

        circuit = QuantumCircuit(num_qubits)
        for gate, pos in gate_sequence:
            circuit.add_gate(gatedict[gate](*pos))
            # reset qubit manually at each Measurement
            if gate == 'Meas':
                circuit.update_quantum_state(state)
                if state.get_classical_value(pos[1]) == 1:
                    circuit = QuantumCircuit(num_qubits)
                    circuit.add_X_gate(pos[0])
                    circuit.update_quantum_state(state)
                circuit = QuantumCircuit(num_qubits)
        # run the remaining circuit after the last Measurement
        if gate != 'Meas':
            circuit.update_quantum_state(state)   

        anc_meas = ''.join([str(state.get_classical_value(i)) for i in range(num_meas)])

        if target_outcomes is None:
            break

    if verbose:
        print(f'ancilla meas: {anc_meas}')
        print_state(state.get_vector(), computational_states)

    return state, anc_meas
    
def get_state_dict(state: QuantumState, computational_states: list[str], num_data: int = 7) -> dict[str, complex]:
    """
    Get a dictionary representation of the quantum state.

    Args:
        state (QuantumState): The quantum state vector.
        computational_states (list[str]): List of computational states.
        num_data (int): Number of data qubits.

    Returns:
        dict[str, complex]: Dictionary representation of the quantum state.
    """
    state_dict = {}
    for i, s in enumerate(computational_states):
        s = s[:num_data]
        if abs(state[i]) > 1e-6:
            state_dict[s] = state[i]
    return state_dict

def check_encoding(gate_sequence: list[tuple[str, tuple[int]]], num_qubits: int, num_meas: int, lut: dict[str, int],
                   correct: str, title: str = '') -> bool:
    """
    Check the encoding of a stabilizer circuit.

    Args:
        gate_sequence (list[tuple[str, tuple[int]]]): List of gates and their positions.
        num_qubits (int): Number of qubits in the circuit.
        num_meas (int): Number of measurement qubits.
        lut (dict[str, int]): Look-up table for ancilla measurements.
        correct (str): Type of correction to apply ('X' or 'Z').
        title (str): Title for the output.

    Returns:
        bool: True if the encoding is correct, False otherwise.
    """
    computational_states = [''.join(map(str,i))[::-1] for i in itertools.product([0, 1], repeat=num_qubits)]

    # get code word
    state, anc_meas = run_stabilizer_circuit(gate_sequence, num_qubits, num_meas, target_outcomes='000')
    codeword = get_state_dict(state.get_vector(),computational_states)

    synd_list = list(lut.keys())

    print(title)
    print('Comparison to logical state before and after correction')
    afters = []
    while len(synd_list) > 0:
        state, anc_meas = run_stabilizer_circuit(gate_sequence, num_qubits, num_meas)
        before = get_state_dict(state.get_vector(),computational_states) == codeword
        # print(anc_meas,synd_list)
        if anc_meas in synd_list:
            synd_list.remove(anc_meas)
        else:
            continue

        # correct single qubit
        if anc_meas != '000':
            circuit = QuantumCircuit(num_qubits)
            if correct == 'Z':
                circuit.add_Z_gate(lut[anc_meas])
            elif correct == 'X':
                circuit.add_X_gate(lut[anc_meas])
            circuit.update_quantum_state(state)
        after = get_state_dict(state.get_vector(),computational_states) == codeword
        afters.append(after)

        print(f'   synd = {anc_meas}:  {before}  ->  {after}')
    return all(afters)

############################## TESTING ##############################

def test_check_encoding():
    test_cases = []

    # standard X-checks: X3456, X1256, X0246
    gate_sequence = [
        ('H', (7,)),
        *[('CX', (control, target)) for control, target in zip([7]*4,[3,4,5,6])],
        ('H', (7,)), 
        ('Meas', (7,0)),

        ('H', (7,)),
        *[('CX', (control, target)) for control, target in zip([7]*4,[1,2,5,6])],
        ('H', (7,)), 
        ('Meas', (7,1)),

        ('H', (7,)),
        *[('CX', (control, target)) for control, target in zip([7]*4,[0,2,4,6])],
        ('H', (7,)), 
        ('Meas', (7,2)),
    ]
    lut = {
        '000': 1000, '100': 3, '010': 1, '001': 0,
        '110': 5, '101': 4, '011': 2, '111': 6,
    }
    test_cases.append([gate_sequence, 8, 3, lut, 'Z', '\n--- standard X-checks -> logical0 state'])

    # standard Z-checks: Z3456, Z1256, Z0246
    gate_sequence = [
        *[('H', (i,)) for i in range(7)],
        *[('CX', (control, target)) for control, target in zip([3,4,5,6],[7]*4)],
        ('Meas', (7,0)),

        *[('CX', (control, target)) for control, target in zip([1,2,5,6],[7]*4)],
        ('Meas', (7,1)),

        *[('CX', (control, target)) for control, target in zip([0,2,4,6],[7]*4)],
        ('Meas', (7,2)),
    ]
    lut = {
        '000': 1000, '100': 3, '010': 1, '001': 0,
        '110': 5, '101': 4, '011': 2, '111': 6,
    }
    test_cases.append([gate_sequence, 8, 3, lut, 'X', '\n--- standard Z-checks -> logical+ state'])
    test_cases.append([gate_sequence + [('H', (i,)) for i in range(7)], 8, 3, lut, 
                       'Z', '\n--- standard Z-checks + Hadamard -> logical0 state'])

    # Z-checks: Z0123, Z2415, Z5623
    gate_sequence = [
        *[('H', (i,)) for i in range(7)],

        ('H', (8,)), ('H', (9,)),
        *[('CX', (control, target)) for control, target in zip([8,9,0,1,2,3,9,8],[7,8,7,7,8,9,8,7])],
        ('H', (8,)), ('H', (9,)),
        ('Meas', (7,0)),

        ('H', (7,)), ('H', (10,)),
        *[('CX', (control, target)) for control, target in zip([7,10,2,4,1,5,10,7],[8,7,8,8,7,10,7,8])],
        ('H', (7,)), ('H', (10,)),
        ('Meas', (8,1)),

        ('H', (8,)), ('H', (9,)),
        *[('CX', (control, target)) for control, target in zip([8,9,5,6,2,3,9,8],[10,8,10,10,8,9,8,10])],
        ('H', (8,)), ('H', (9,)),
        ('Meas', (10,2)),
    ]
    lut = {
        '000': 1000, '100': 0, '010': 4, '001': 6,
        '110': 1, '101': 3, '011': 5, '111': 2,
    }
    test_cases.append([gate_sequence, 11, 3, lut, 'X', '\n--- 2-flag ancilla-centered Z-checks with CX -> logical+ state'])
    test_cases.append([gate_sequence + [('H', (i,)) for i in range(7)], 11, 3, lut, 
                       'Z', '\n--- 2-flag ancilla-centered Z-checks with CX +  Hadamard -> logical0 state'])

    run_test([test_cases, [True]*len(test_cases)], lambda x: check_encoding(*x), 'check_encoding')

def test_all():
    print(f'\nTesting functions in {os.path.basename(__file__)} ...\n')
    test_check_encoding()
    print()
    print()


if __name__ == "__main__":
    test_all()