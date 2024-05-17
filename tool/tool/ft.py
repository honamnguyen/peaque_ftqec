import os
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
    for i, (faulty_gate, faulty_locs) in enumerate(gate_seq):
        num_locs = len(faulty_locs)
        faults = get_faults(fault_types, weight1_only)[num_locs-1]
        for fault in faults:
            # get the starting fault string
            fault_string = get_fault_string(num_qubits, faulty_locs, fault)
            # update the fault string with the subsequent gates
            for gate, position in gate_seq[i:]:
                fault_string = qec.clifford_transform(list(fault_string), gate, position)

            final_error = ''.join(fault_string[:num_datas]) + '|' + ''.join(fault_string[num_datas:])
            all_locs.append([i, gate_seq[i], fault, final_error])
            if qec.pauli_weight(fault_string[:num_datas]) > 1:
                bad_locs.append([i, gate_seq[i], fault, final_error])

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

# def print_locations(locations, remove_z=False, ancilla_outcomes=None):
#     """
#     Print the locations of faults and the final errors.

#     Args:
#         locations (List): List of locations.
#     """
#     if len(locations) == 0:
#         print('No bad locations')
#         return
#     first_line = ' idx  gate  location  fault    final_error'
#     if len(locations[0]) > 4: first_line += '  equiv_error  weight'
#     if remove_z: first_line += '  remove_Zerr  weight'
#     if ancilla_outcomes is not None: 
#         assert len(ancilla_outcomes) == len(locations)
#         first_line += '    ancilla_outcomes'
#         anc_out_len = max(len(ancilla_outcomes[0]),len('ancilla_outcomes')) + 6
#     print(first_line)
#     for i,loc in enumerate(locations):
#         print_str = str(loc[0]).center(5)
#         print_str += str(loc[1][0]).center(6)
#         gate_loc = [j for j in loc[1][1]]
#         print_str += str(gate_loc).center(10)
#         print_str += str(loc[2]).center(7)
#         print_str += str(loc[3]).center(15)
#         if len(locations[0]) > 4:
#             print_str += str(loc[4]).center(13)
#             print_str += str(loc[5]).center(8)
#         if remove_z:
#             print_str += str(loc[6]).center(13)
#             print_str += str(loc[7]).center(8)
#         if ancilla_outcomes is not None:
#             print_str += str(ancilla_outcomes[i]).center(anc_out_len)

#         print(print_str)

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

def test_all():

    print(f'\nTesting functions in {os.path.basename(__file__)} ...\n')
    test_get_faults()
    test_get_fault_string()
    test_get_bad_locations()
    test_update_locations()
    test_reset_ancillas()
    test_modulo_stabilizers()
    test_remove_z_errors()
    print()
    print()


if __name__ == "__main__":
    test_all()