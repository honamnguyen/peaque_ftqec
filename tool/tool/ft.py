import os
import numpy as np
import itertools
from typing import List, Tuple
from tool import qec
from tool.testing import run_test

def get_faults(fault_types: str, weight1_only: bool = False) -> List[List[str]]:
    """
    Get all possible faults based on the given fault types.

    Args:
        fault_types (str): String containing the fault types.
        weight1_only (bool, optional): Whether to include only weight-1 faults. Defaults to False.

    Returns:
        List[List[str]]: List of all possible faults.
    """
    all_faults = [list(fault_types), []]
    if weight1_only:
        for fault_type in fault_types:
            all_faults[-1] += [''.join(f) for f in itertools.permutations('-' + fault_type, 2)]
    else:
        all_faults[-1] += [''.join(f) for f in itertools.product('-' + fault_types, repeat=2)]
        all_faults[-1].remove('--')
    return all_faults

def get_fault_string(num_qubits: int, locs: List, fault: str) -> List[str]:
    """
    Get the fault string based on the number of qubits, locations, and fault type.

    Args:
        num_qubits (int): Number of qubits.
        locs (List): List of locations.
        fault (str): Fault type.

    Returns:
        List[str]: Fault string.

    Example: 
        >>> ''.join(get_fault_string(5,[1,3],'XZ')) 
        '-X-Z-'
        >>> ''.join(get_fault_string(5,[4,0],'YX')) 
        'X---Y'
    """
    fault_string = ['-'] * num_qubits
    for i in range(len(locs)):
        fault_string[locs[i]] = fault[i]
    return fault_string    

def get_bad_locations(
    gate_seq: List, 
    fault_types: str, 
    num_qubits: int, 
    num_datas: int, 
    weight1_only: bool = 'False', 
    verbose: str = 'bad locations',
) -> List:
    """
    Get the bad locations based on the gate sequence, fault types, and other parameters.
    Faults are inserted AFTER each gate in the gate sequence.

    Args:
        gate_seq (List): List of gate sequences.
        fault_types (str): String containing the fault types.
        weight1_only (bool): Whether to include only weight-1 faults.
        num_qubits (int): Number of qubits.
        num_datas (int): Number of data qubits.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        List: List of bad locations.
    """
    all_locs = []
    bad_locs = []
    
    gate_seq = [('I', (j,)) for j in range(num_qubits)] + list(gate_seq)
    # loop through initial idle errors then the gate sequence
    for i, (faulty_gate, faulty_locs) in enumerate(gate_seq):
        num_locs = len(faulty_locs)
        faults = get_faults(fault_types, weight1_only)[num_locs-1]
        for fault in faults:
            # get the starting fault string
            fault_string = get_fault_string(num_qubits, faulty_locs, fault)
            # update the fault string with the subsequent gates
            for gate, position in gate_seq[i+1:]:
                fault_string = qec.clifford_transform(list(fault_string), gate, position)

            final_error = ''.join(fault_string[:num_datas]) + '|' + ''.join(fault_string[num_datas:])
            all_locs.append([max(i-num_qubits,-1), gate_seq[i], fault, final_error])
            if qec.pauli_weight(fault_string[:num_datas]) > 1:
                bad_locs.append([max(i-num_qubits,-1), gate_seq[i], fault, final_error])

    if verbose == 'bad locations':
        print(' idx  gate  location  fault  final_error')
        for loc in bad_locs:
            print_str = str(loc[0]).center(5)
            print_str += str(loc[1][0]).center(6)
            print_str += str(loc[1][1]).center(10)
            print_str += str(loc[2]).center(6)
            print_str += str(loc[3]).center(13)
            print(print_str)
    elif verbose == 'all locations':
        print(' idx  gate  location  fault  final_error')
        for loc in all_locs:
            print_str = str(loc[0]).center(5)
            print_str += str(loc[1][0]).center(6)
            print_str += str(loc[1][1]).center(10)
            print_str += str(loc[2]).center(6)
            print_str += str(loc[3]).center(13)
            print(print_str)
    return bad_locs, all_locs

def update_locations(locations: List, gate_sequence: Tuple, num_datas: int) -> List:
    """
    Updates the locations based on the gate sequence.

    Args:
        locations (List): List of locations.
        gate_sequence (Tuple): Gate sequence.
        num_datas (int): Number of data qubits.

    Returns:
        List: Updated list of locations.
    """
    updated_locs = []
    for loc in locations:
        fault_string = list(''.join(loc[-1].split('|')))
        for gate, position in gate_sequence:
            fault_string = qec.clifford_transform(list(fault_string), gate, position)
        final_error = ''.join(fault_string[:num_datas]) + '|' + ''.join(fault_string[num_datas:])
        updated_locs.append(loc[:-1] + [final_error])
    return updated_locs

def reset_ancillas(locations: List) -> Tuple[List, List]:
    """
    Reset the ancilla qubits in the given locations.

    Args:
        locations (List[str]): List of locations containing ancilla qubits.

    Returns:
        Tuple[List, List]: Updated locations with resetted ancillas and ancillas measurment outcomes.
    """
    triggered = {'X': 1, 'Y': 1, 'Z': 0, '-': 0}
    updated_locations = []
    outcomes = []    
    for loc in locations:
        datas, ancillas = loc[-1].split('|')
        outcome = [triggered[p] for p in ancillas]
        outcomes.append(outcome)
        updated_locations.append(loc[:-1] + [datas + '|' + '-'*len(ancillas)])
    return updated_locations, outcomes

def modulo_stabilizers(bad_locations, stabilizer_group, return_inds=False):
    """
    Modulo the bad locations with the stabilizer group to find the
    smallest-weight equivalent errors.

    Args:
        bad_locations (List): List of bad locations.
        stabilizer_group: Stabilizer group.

    Returns:
        Tuple: Tuple of 
            updated bad locations: List of original bad locations with the smallest-weight equivalent errors.
            remaining bad locations: List of bad locations with weight > 1.
    """
    updated_bad_locations = []
    remaining_bad_locations = []
    remaining_bad_inds = []
    for i,loc in enumerate(bad_locations):
        error = list(loc[-1].split('|')[0])
        equiv_error, weight = qec.lowest_weight_equivalent(error, stabilizer_group)
        updated_bad_locations.append(tuple(list(loc) + [''.join(equiv_error), weight]))
        if weight > 1:
            remaining_bad_locations.append(updated_bad_locations[-1])
            remaining_bad_inds.append(i)
    if return_inds:
        return updated_bad_locations, remaining_bad_locations, remaining_bad_inds
    else:
        return updated_bad_locations, remaining_bad_locations

def remove_z_errors(locations: List[Tuple]) -> List[Tuple]:
    '''
    Remove Z errors from the locations list
    
    Args:
        locations (List[Tuple[int, Tuple[str, Tuple[int]]]]): List of locations
        
    Returns:
        List[Tuple[int, Tuple[str, Tuple[int]]]]: Updated list of locations with Z errors removed
    '''
    remove_z = {'X':'X', 'Y':'X', 'Z':'-', '-':'-'}
    updated_locations = []
    remained_locations = []
    for loc in locations:
        updated_error = ''
        for p in loc[-2]:
            updated_error += remove_z[p]
        weight = qec.pauli_weight(updated_error)
        updated_locations.append(tuple(list(loc) + [updated_error, weight]))
        if weight > 1:
            remained_locations.append(updated_locations[-1])
    return updated_locations, remained_locations

def remove_x_errors(locations: List[Tuple]) -> List[Tuple]:
    '''
    Remove X errors from the locations list
    
    Args:
        locations (List[Tuple[int, Tuple[str, Tuple[int]]]]): List of locations
        
    Returns:
        List[Tuple[int, Tuple[str, Tuple[int]]]]: Updated list of locations with Z errors removed
    '''
    remove_x = {'X':'-', 'Y':'Z', 'Z':'Z', '-':'-'}
    updated_locations = []
    remained_locations = []
    for loc in locations:
        updated_error = ''
        for p in loc[-2]:
            updated_error += remove_x[p]
        weight = qec.pauli_weight(updated_error)
        updated_locations.append(tuple(list(loc) + [updated_error, weight]))
        if weight > 1:
            remained_locations.append(updated_locations[-1])
    return updated_locations, remained_locations

def print_locations(locations, extras=None):
    """
    Print the locations of faults and the final errors.

    Args:
        locations (List): List of locations.
    """
    if len(locations) == 0:
        print('No bad locations')
        return
    first_line = ' idx  gate  location  fault   final_error'
    if extras is not None:
        for label, _ in extras:
            first_line += label
    print(first_line)

    for i,loc in enumerate(locations):
        print_str = str(loc[0]).center(5)
        print_str += str(loc[1][0]).center(6)
        gate_loc = [j for j in loc[1][1]]
        print_str += str(gate_loc).center(10)
        print_str += str(loc[2]).center(7)
        print_str += str(loc[3]).center(15)
        if extras is not None:
            for i, (label, center_len) in enumerate(extras):
                first_line += label
                print_str += str(loc[4+i]).center(center_len)

        print(print_str)

def run_sequences(sequences: List[str], bad_locations_only: bool = False) -> Tuple[List, List]:
    """
    Run sequences of gates with a single fault in the first sequence.
    Return the propagated errors and ancilla outcomes from all faults.

    Args:
        sequences (List[str]): List of gate sequences.
        bad_locations_only (bool): Indicator for returning only the bad locations.

    Returns:
        Tuple[List, List]: Tuple of 
            locations: List of locations.
            ancilla_outcomes: List of ancilla outcomes.
    """
    bad_locations, all_locations = get_bad_locations(
        qec.get_sequence(sequences[0]), 'XYZ', 11, 7, 
        weight1_only=False, verbose=''
    )
    
    if bad_locations_only:
        locations = bad_locations
    else:
        locations = all_locations

    ancilla_outcomes = []
    locations, outcomes = reset_ancillas(locations)
    ancilla_outcomes.append([''.join(map(str,out)) for out in outcomes])

    for sequence in sequences[1:]:
        locations = update_locations(locations, qec.get_sequence(sequence), 7)
        locations, outcomes = reset_ancillas(locations)
        ancilla_outcomes.append([''.join(map(str,out)) for out in outcomes])

    ancilla_outcomes = ['||'.join(out) for out in np.array(ancilla_outcomes).T]

    return locations, ancilla_outcomes

look_up_table = {
    'Steane_flag_bridge_SZ': {
        '000': '-------',
        '100': 'X------',
        '010': '----X--',
        '110': '-X-----',
        '111': '--X----',
        '001': '------X',
        '101': '---X---',
        '011': '-----X-',
        
        '00': '-------',
        '10': '----X--',
        '01': '------X',
        '11': '-----X-',
                
        '0': '-------',
        '1': '------X',
    },

    'Steane_flag_bridge_SX': {
        '000': '-------',
        '100': 'Z------',
        '010': '----Z--',
        '110': '-Z-----',
        '111': '--Z----',
        '001': '------Z',
        '101': '---Z---',
        '011': '-----Z-',
        
        '00': '-------',
        '10': '----Z--',
        '01': '------Z',
        '11': '-----Z-',
        
        '0': '-------',
        '1': '------Z',
    }

}
def process_ancillas(locations: List, ancilla_outcomes: List, synd_inds: List[int]) -> Tuple[List, List]:
    """
    Process the ancilla outcomes and return the updated locations with information about syndromes and flags.

    Args:
        locations (List): List of locations.
        ancilla_outcomes (List): List of ancilla outcomes.
        synd_inds (List[int]): List of syndrome indices.

    Returns:
        Tuple[List, List]: Tuple of 
            updated_locations: List of updated locations.
            syndromes: List of syndromes.
    """
    flags = []
    syndromes = []
    assert len(ancilla_outcomes[0].split('||')) == len(synd_inds)
    for outcome in ancilla_outcomes:
        ancillas = outcome.split('||')
        syndromes.append(''.join([anc[synd_inds[i]] for i,anc in enumerate(ancillas)]))
        flag = ''.join([anc[:synd_inds[i]]+anc[synd_inds[i]+1:] for i,anc in enumerate(ancillas)])
        if '1' in flag:
            flags.append(1)
        else:
            flags.append(0)

    updated_locations = []
    for i in range(len(syndromes)):
        updated_locations.append(locations[i] + 
                                 [ancilla_outcomes[i], syndromes[i], flags[i]] + 
                                 [locations[i][-1].split('|')[0]])
    return updated_locations, syndromes

def correct_errors(locations: List, syndromes: List, lut: dict) -> List:
    """
    Correct errors in the locations based on the syndromes and look-up table.

    Args:
        locations (List): List of locations.
        syndromes (List): List of syndromes.
        lut (dict): Look-up table.

    Returns:
        List: List of updated locations.
    """
    updated_locations = []
    for i in range(len(syndromes)):
        error = locations[i][-1]
        corrected_error = qec.compose_two_paulis(error,list(lut[syndromes[i]]))
        updated_locations.append(locations[i] + [''.join(corrected_error)])
    return updated_locations

def check_ft(sequences: List[str], synd_inds: List[int], lut_name: str, stabilizer_group, verbose: int = 2) -> bool:
    """
    Check the fault tolerance of gate sequences.

    Args:
        sequences (List[str]): List of gate sequences, each is a stabilizer check circuit.
        synd_inds (List[int]): List of syndrome indices.
        lut_name (str): Look-up table name.
        stabilizer_group: Stabilizer group.
        verbose (int):
            0: No output.
            1: Print only the harmful errors.
            2: Print all errors.

    Returns:
        bool: True if the fault tolerance is satisfied, False otherwise.
    """
    locations, ancilla_outcomes =  run_sequences(sequences, bad_locations_only=False)
    print_extras = []
    # process ancilla outcomes
    locations, syndromes = process_ancillas(locations, ancilla_outcomes, synd_inds)
    print_extras += [('    ancilla_outcomes', max(len(ancilla_outcomes[0]),16) + 4)]
    print_extras += [('  synds',6), ('  flag',6),  ('   final ',9)]

    # correct errors
    locations = correct_errors(locations, syndromes, look_up_table[lut_name])
    print_extras += [('  corrected', 12)]

    # update bad locations when modulo the stabilizer group
    locations, _, _ = modulo_stabilizers(locations, stabilizer_group, True)
    print_extras += [('  equiv_error',12), ('  wt',6)]

    if lut_name[-1] == 'Z':
        # remove X errors
        locations, remaining_locations = remove_x_errors(locations)
        print_extras += [('  remove_Xerr',10), ('  wt',8)]
    elif lut_name[-1] == 'X':
        # remove Z errors
        locations, remaining_locations = remove_z_errors(locations)
        print_extras += [('  remove_Zerr',10), ('  wt',8)]

    if verbose > 0:
        ent_gate, stab = sequences[0].split('_')[-2:]
        print(f'\n------All harmful errors from {stab} circuit with {ent_gate}------')
        print_locations(remaining_locations, print_extras)

    ################ check flags and syndromes ################
    unflagged_locations = []
    flagged_synd0_locations = []
    flagged_synd1_locations = []

    unflagged_weight2 = False
    for loc in locations:
        if loc[6] == 1:
            if '1' in loc[5]:
                flagged_synd1_locations.append(loc)
            else:
                flagged_synd0_locations.append(loc)
        else:
            unflagged_locations.append(loc)
            if loc[-1] > 1:
                unflagged_weight2 = True
    assert unflagged_weight2 == False

    if verbose > 1:
        print('\n------Unflagged locations (accepted)------')
        print_locations(unflagged_locations, print_extras)
        print('-----Unflagged weight > 1:',unflagged_weight2)

        print('\n------Flagged locations with synd=0 (rejected)------')
        print_locations(flagged_synd0_locations, print_extras)

        print('\n------Flagged locations with synd=1 (rejected)------')
        print_locations(flagged_synd1_locations, print_extras)

    return not unflagged_weight2


############################## TESTING ##############################
def test_get_faults():
    """
    Test the get_faults function.
    """
    test_cases = {
            # ('Y', False): [['Z'], ['-Z', 'ZX', 'ZZ']], # incorrect test for testing only
            ('Z', False): [['Z'], ['-Z', 'Z-', 'ZZ']],
            ('Z', True): [['Z'], ['-Z', 'Z-']],
            ('XYZ', False): [['X', 'Y', 'Z'], ['-X', '-Y', '-Z', 'X-', 'XX', 'XY', 'XZ', 'Y-', 'YX', 'YY', 'YZ', 'Z-', 'ZX', 'ZY', 'ZZ']],
            ('XYZ', True): [['X', 'Y', 'Z'], ['-X', 'X-', '-Y', 'Y-', '-Z', 'Z-']],
        }
    run_test(test_cases, lambda input: get_faults(*input), 'get_faults')
    
def test_get_fault_string():
    """
    Test the get_fault_string function.
    """
    test_cases = {
        (5, (1, 3), 'XZ'): '-X-Z-',
        (5, (4, 0), 'YX'): 'X---Y',
    }
    run_test(test_cases, lambda input: ''.join(get_fault_string(*input)), 'get_fault_string')

def test_get_bad_locations():
    """
    Test the get_bad_locations function.
    """
    print('>> get_bad_locations - No Test <<')
    pass

def test_update_locations():
    """
    Test the update_locations function.
    """
    print('>> update_locations - No Test <<')
    pass

def test_reset_ancillas():
    """
    Test the reset_ancillas function.
    """
    print('>> reset_ancillas - No Test <<')
    pass

def test_modulo_stabilizers():
    """
    Test the modulo_stabilizers function.
    """
    print('>> modulo_stabilizers - No Test <<')
    pass

def test_remove_z_errors():
    """
    Test the remove_z_errors function.
    """
    print('>> remove_z_errors - No Test <<')
    pass

def test_check_ft():
    """
    Test the check_ft function.
    """

    test_cases = []

    # Steane code:
    stabilizer_generators = ['ZZZZ---','-ZZ-ZZ-','--ZZ-ZZ',
                         'XXXX---','-XX-XX-','--XX-XX'] #, 'XXXXXXX']
    stabilizer_generators = [list(stab) for stab in stabilizer_generators]
    stabilizer_group = qec.compute_stabilizer_group(stabilizer_generators)

    # 3 SZ plaquettes with CX
    test_cases.append([['flag_bridge_CX_SZ1','flag_bridge_CX_SZ2','flag_bridge_CX_SZ3'],
                       [0,1,3], 'Steane_flag_bridge_SZ', stabilizer_group])
    test_cases.append([['flag_bridge_CX_SZ2','flag_bridge_CX_SZ3'],
                       [1,3], 'Steane_flag_bridge_SZ', stabilizer_group])
    test_cases.append([['flag_bridge_CX_SZ3'],
                       [3], 'Steane_flag_bridge_SZ', stabilizer_group])
    # 3 SX plaquettes with CX    
    test_cases.append([['flag_bridge_CX_SX1','flag_bridge_CX_SX2','flag_bridge_CX_SX3'],
                       [0,1,3], 'Steane_flag_bridge_SX', stabilizer_group])
    test_cases.append([['flag_bridge_CX_SX2','flag_bridge_CX_SX3'],
                       [1,3], 'Steane_flag_bridge_SX', stabilizer_group])
    test_cases.append([['flag_bridge_CX_SX3'],
                       [3], 'Steane_flag_bridge_SX', stabilizer_group])
    # 3 SZ plaquettes with CZ
    test_cases.append([['flag_bridge_CZ_SZ1','flag_bridge_CZ_SZ2','flag_bridge_CZ_SZ3'],
                       [0,1,3], 'Steane_flag_bridge_SZ', stabilizer_group])
    test_cases.append([['flag_bridge_CZ_SZ2','flag_bridge_CZ_SZ3'],
                       [1,3], 'Steane_flag_bridge_SZ', stabilizer_group])
    test_cases.append([['flag_bridge_CZ_SZ3'],
                       [3], 'Steane_flag_bridge_SZ', stabilizer_group])
    # 3 SX plaquettes with CZ
    test_cases.append([['flag_bridge_CZ_SX1','flag_bridge_CZ_SX2','flag_bridge_CZ_SX3'],
                       [0,1,3], 'Steane_flag_bridge_SX', stabilizer_group])
    test_cases.append([['flag_bridge_CZ_SX2','flag_bridge_CZ_SX3'],
                       [1,3], 'Steane_flag_bridge_SX', stabilizer_group])
    test_cases.append([['flag_bridge_CZ_SX3'],
                       [3], 'Steane_flag_bridge_SX', stabilizer_group])

    run_test([test_cases, [True]*len(test_cases)], lambda x: check_ft(*x, verbose=0), 'check_ft')
    

def test_all():

    print(f'\nTesting functions in {os.path.basename(__file__)} ...\n')
    test_get_faults()
    test_get_fault_string()
    test_get_bad_locations()
    test_update_locations()
    test_reset_ancillas()
    test_modulo_stabilizers()
    test_remove_z_errors()
    test_check_ft()
    print()
    print()


if __name__ == "__main__":
    test_all()