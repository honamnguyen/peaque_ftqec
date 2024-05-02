import numpy as np
from typing import List, Tuple
import itertools

from tool.testing import run_test

"""
A module that provides various quantum error correction tools.

Methods:
    compose_paulis(ps: List[List[str]]) -> List[str]: Composes multiple Pauli strings element-wise.
    compose_two_paulis(p1: List[str], p2: List[str]) -> List[str]: Composes two Pauli strings element-wise.
    clifford_transform(pauli_string: List[str], gate: str, position: List[int]) -> List[str]: Applies a Clifford gate to a Pauli string at specified positions.
    pauli_weight(p_str): Compute the weight of a Pauli string.
    pauli_display(ps): Display Pauli strings.
    common_gate(gatename): Get the gate matrix based on the gate name.
    common_qecc(name): Returns a list of stabilizers for a given quantum error correcting code (QECC).
    compute_stabilizer_group(stabilizer_generators: List[List[str]]) -> List[List[str]]: Compute the full stabilizer group given a list of stabilizer generators.
    test_all(): Runs all the test methods.
"""

def compose_paulis(ps: List[List[str]]) -> List[str]:
    """
    Composes multiple Pauli strings element-wise.

    Args:
        ps (List[List[str]]): List of Pauli strings.

    Returns:
        List[str]: Composed Pauli string.
    """
    out_p = ps[0]
    for i in range(1, len(ps)):
        out_p = compose_two_paulis(out_p, ps[i])
    return out_p

def compose_two_paulis(p1: List[str], p2: List[str]) -> List[str]:
    """
    Composes two Pauli strings element-wise.

    Args:
        p1 (List[str]): First Pauli string.
        p2 (List[str]): Second Pauli string.

    Returns:
        List[str]: Composed Pauli string.
    """
    out_p = ['-'] * len(p1)
    for i in range(len(p1)):
        if p1[i] == '-':
            out_p[i] = p2[i]
        elif p2[i] == '-':
            out_p[i] = p1[i]
        elif p1[i] == p2[i]:
            out_p[i] = '-'
        else:
            for pauli in ['X', 'Y', 'Z']:
                if pauli not in [p1[i], p2[i]]:
                    out_p[i] = pauli
    return out_p


def clifford_transform(pauli_string: List[str], gate: str, position: List[int]) -> List[str]:
    """
    Applies a Clifford gate to a Pauli string at specified positions.

    Args:
        pauli_string (List[str]): Pauli string.
        gate (str): Clifford gate.
        position (List[int]): Positions to apply the gate.

    Returns:
        List[str]: Transformed Pauli string.
    """
    pauli_string = pauli_string
    if len(position) == 1:
        current = pauli_string[position[0]]
        new = clifford_transform_dict[gate][current]   
        pauli_string[position[0]] = new
    elif len(position) == 2:
        current = pauli_string[position[0]] + pauli_string[position[1]]
        new = clifford_transform_dict[gate][current]
        pauli_string[position[0]] = new[0]
        pauli_string[position[1]] = new[1]
    return pauli_string

def clifford_transform_sequence(pauli_string: List[str], sequence: List) -> List[str]:
    """
    Applies a sequence of Clifford gates to a Pauli string.

    Args:
        pauli_string (List[str]): Pauli string.
        sequence (List): List of tuples containing the gate and position.

    Returns:
        List[str]: Transformed Pauli string.
    """
    for gate, position in sequence:
        pauli_string = clifford_transform(pauli_string, gate, position)
    return pauli_string


clifford_transform_dict = {
    'H': {
        '-': '-',
        'X': 'Z',
        'Y': 'Y',
        'Z': 'X',
    },
    'S': {
        '-': '-',
        'X': 'Y',
        'Y': 'X',
        'Z': 'Z',
    },
    'CX': {
        '--': '--',
        '-X': '-X',
        '-Y': 'ZY',
        '-Z': 'ZZ',

        'X-': 'XX',
        'XX': 'X-',
        'XY': 'YZ',
        'XZ': 'YY',

        'Y-': 'YX',
        'YX': 'Y-',
        'YY': 'XZ',
        'YZ': 'XY',
        
        'Z-': 'Z-',
        'ZX': 'ZX',
        'ZY': '-Y',
        'ZZ': '-Z',
    },
    'CZ': {
        '--': '--',
        '-X': 'ZX',
        '-Y': 'ZY',
        '-Z': '-Z',

        'X-': 'XZ',
        'XX': 'YY',
        'XY': 'YX',
        'XZ': 'X-',

        'Y-': 'YZ',
        'YX': 'XY',
        'YY': 'XX',
        'YZ': 'Y-',

        'Z-': 'Z-',
        'ZX': '-X',
        'ZY': '-Y',
        'ZZ': 'ZZ',
    }
}
def pauli_weight(p_str):
    """
    Compute the weight of a Pauli string.

    Args:
        p_str (List or str): Pauli string.

    Returns:
        int: Weight of the Pauli string.
    """
    return sum([p != '-' for p in p_str])

def pauli_display(ps):
    """
    Display Pauli strings.

    Args:
        ps (List): a Pauli string or a list of Pauli strings.
    """
    if type(ps[0]) == str:
        print(''.join(ps))
    elif type(ps[0]) == list:
        for p in ps:
            print(''.join(p))    

def common_gate(gatename):
    """
    Get the gate matrix based on the gate name.

    Args:
        gatename (str): Gate name.

    Returns:
        numpy.ndarray: Gate matrix.
    """
    gate_dict = {
        '-': np.eye(2, dtype=np.int32),
        'X': np.array([[0, 1], [1, 0]], dtype=np.int32),
        'Y': np.array([[0, -1j], [1j, 0]], dtype=np.complex64),
        'Z': np.array([[1, 0], [0, -1]], dtype=np.int32),
        'H': np.array([[1, 1], [1, -1]], dtype=np.int32) / np.sqrt(2),
        'S': np.array([[1, 0], [0, 1j]], dtype=np.complex64),
        'CX': np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.int32),
    }
    return gate_dict[gatename]

def common_qecc(name) -> Tuple[str]:
    """
    Returns a list of stabilizers for a given quantum error correcting code (QECC).

    Parameters:
    name (str): The name of the QECC. Currently supported QECCs are 'bitflip_code' and 'steane_code_Goto'.

    Returns:
    list: A list of stabilizers for the specified QECC.

    Raises:
    KeyError: If the specified QECC name is not found in the qecc_dict.

    Example:
    >>> common_qecc('bitflip_code')
    ['ZZ-', '-ZZ']
    >>> common_qecc('steane_code_Goto')
    ['---ZZZZ', 'ZZ--ZZ-', 'Z-Z-Z-Z', '---XXXX', 'XX--XX-', 'X-X-X-X']
    """
    
    qecc_dict = {
        'bitflip_code' :  ('ZZ-', '-ZZ'),
        'steane_code_Goto' : ('---ZZZZ', 'ZZ--ZZ-', 'Z-Z-Z-Z', '---XXXX', 'XX--XX-', 'X-X-X-X'),
    }
    # return [list(stabilizer) for stabilizer in qecc_dict[name]]
    return qecc_dict[name]

def get_sequence(name: str) -> Tuple[Tuple[str, Tuple[int]]]:
    """
    Returns a sequence of Clifford gates based on the specified name.

    Args:
        name (str): Name of the sequence.

    Returns:
        Tuple[Tuple[str, Tuple[int]]]: Sequence of Clifford gates.

    References:
        Goto (efficient Steane encoding): https://www.nature.com/articles/srep19578
        Aliferis, Gottesman, Preskill: https://arxiv.org/abs/quant-ph/0504218
            - Cat encoding (DiVinzenzo-Shor): Fig. 6

    """
    sequence_dict = {
        'Goto_1c': (
            ('H', (1,)), ('H', (2,)), ('H', (3,)),
            ('CX', (1, 0)), ('CX', (3, 5)),
            ('CX', (2, 6)),
            ('CX', (1, 4)),
            ('CX', (2, 0)), ('CX', (3, 6)),
            ('CX', (1, 5)),
            ('CX', (6, 4)),
        ),
        'ZZZZ_nonFT': (
            ('CX', (0, 4)),
            ('CX', (1, 4)),
            ('CX', (2, 4)),
            ('CX', (3, 4))
        ),
        'cat_encoding_divicenzoshor': (
            ('H', (1,)),
            ('CX', (1, 2)),
            ('CX', (1, 0)), ('CX', (2, 3)),
        ),
        # measurement is currently in +/- so need to add Hadamard on last 3 qubits
        'flag_bridge_CZ_single': (
            ('H', (4,)), ('H', (5,)), ('H', (6,)),
            ('CZ', (4, 5)),
            ('H', (5,)),
            ('CZ', (5, 6)),
            ('CZ', (0, 4)),
            ('CZ', (1, 4)),
            ('CZ', (2, 5)), ('H', (6,)),
            ('CZ', (3, 6)),
            ('H', (6,)),
            ('CZ', (5, 6)),
            ('H', (5,)),
            ('CZ', (4, 5)),
            ('H', (4,)), ('H', (5,)), ('H', (6,)),
        ),
        'flag_bridge_CZ_SX1': (
            ('H', (7,)), ('H', (8,)), ('H', (9,)),
            ('CZ', (7, 8)),
            ('H', (8,)),
            ('CZ', (8, 9)), ('CZ', (0, 7)),
            ('CZ', (1, 7)),
            ('CZ', (2, 8)), ('H', (9,)),
            ('CZ', (3, 9)),
            ('H', (9,)),
            ('CZ', (8, 9)),
            ('H', (8,)),
            ('CZ', (7, 8)),
            ('H', (7,)), ('H', (8,)), ('H', (9,)),
        ),
        'flag_bridge_CZ_SX2': (
            ('H', (8,)), ('H', (7,)), ('H', (9,)),
            ('CZ', (8, 7)),
            ('H', (7,)),
            ('CZ', (7, 9)), ('CZ', (2, 8)),
            ('CZ', (4, 8)),
            ('CZ', (3, 7)), ('H', (9,)),
            ('CZ', (5, 9)),
            ('H', (9,)),
            ('CZ', (7, 9)),
            ('H', (7,)),
            ('CZ', (8, 7)),
            ('H', (8,)), ('H', (7,)), ('H', (9,)),
        ),
        'flag_bridge_CZ_SX3': (
            ('H', (10,)), ('H', (8,)), ('H', (9,)),
            ('CZ', (10, 8)),
            ('H', (8,)),
            ('CZ', (8, 9)), ('CZ', (5, 10)),
            ('CZ', (6, 10)),
            ('CZ', (2, 8)), ('H', (9,)),
            ('CZ', (3, 9)),
            ('H', (9,)),
            ('CZ', (8, 9)),
            ('H', (8,)),
            ('CZ', (10, 8)),
            ('H', (10,)), ('H', (8,)), ('H', (9,)),
        ),
    }
    return sequence_dict[name]

def compute_stabilizer_group(stabilizer_generators: List[List[str]]) -> List[List[str]]:
    """
    Compute the full stabilizer group given a list of stabilizer generators.

    Args:
        stabilizer_generators (List[List[str]]): List of stabilizer generators.

    Returns:
        List[List[str]]: List of stabilizer group elements.

    Example:
        >>> stabilizer_generators = [['Z', 'Z', '-'], ['-', 'Z', 'Z']]
        >>> compute_stabilizer_group(stabilizer_generators)
        [['Z', 'Z', '-'], ['-', 'Z', 'Z'], ['X', 'X', 'X'], ['Z', '-', 'Z'], ['Y', 'Y', 'X'], ['X', 'Y', 'Y'], ['Y', 'X', 'Y']]
    """
    stabilizer_group = stabilizer_generators.copy()
    
    # Iterate over the number of stabilizers to compose
    for num_stabs in range(2, len(stabilizer_generators) + 1):
        # Generate all combinations of stabilizer generators
        for stabilizer_list in itertools.combinations(stabilizer_generators, num_stabs):
            group_element = compose_paulis(stabilizer_list)
            if group_element not in stabilizer_group:
                stabilizer_group.append(group_element)
    
    return stabilizer_group

def lowest_weight_equivalent(error: List[str], stabilizer_group: List[List[str]]) -> Tuple[List[str], int]:
    """
    Find the lowest weight equivalent error under a stabilizer group.

    Args:
        error (List[str]): The error to find the lowest weight equivalent for.
        stabilizer_group (List[List[str]]): The stabilizer group to search for the lowest weight equivalent.

    Returns:
        Tuple[List[str], int]: The lowest weight equivalent error and its weight.
    """
    min_weight = pauli_weight(error)
    min_weight_error = error
    for elem in stabilizer_group:
        equiv_error = compose_two_paulis(elem, error)
        weight = pauli_weight(equiv_error)
        if weight < min_weight:
            min_weight = weight
            min_weight_error = equiv_error
    return min_weight_error, min_weight

############################## TESTING ##############################

def test_clifford_transform_dict():

    test_cases = {}
    for gate in ['CX', 'CZ']:
        for p in itertools.product(['-','X','Y','Z'],repeat=2):
            error = ''.join(p)
            if pauli_weight(error) == 2:
                first = clifford_transform_dict['CZ'][error[0]+'-']
                second = clifford_transform_dict['CZ']['-'+error[1]]
                test_cases[error] = ''.join(compose_two_paulis(first,second))
            else:
                test_cases[error] = ''.join(clifford_transform_dict['CZ'][error])
    run_test(test_cases, lambda input: ''.join(clifford_transform_dict['CZ'][input]), 'clifford_transform_dict')


def test_compose_paulis():
    """
    Tests the compose_paulis method.
    """
    test_cases = {
        # ('ZZZXZZZ', 'ZZ--ZZ-'): '--XZX-Z', # incorrect test for testing only
        # ('XYZX-X', '-ZZYXZ'): 'XX-XZY',  # incorrect test for testing only
        ('ZZZZ', 'ZZZZ'): '----',
        ('ZZZZZZZ', 'ZZ--ZZ-'): '--ZZ--Z',
        ('XYZZ-X', '-ZZYXZ'): 'XX-XXY',
        ('XYZZ-X', '-ZZYXZ', 'Y-XZZ-'): 'ZXXYYY',
    }
    run_test(test_cases, lambda input: ''.join(compose_paulis([list(p) for p in input])), 'compose_paulis')

def test_clifford_transform_sequence():
    """
    Tests the clifford_transform method.
    """
    test_cases = {
        # ( '----Z', ( ('CX',(1,4)) , ('CX',(0,2)) ) ) : 'ZX--Z',  # incorrect test for testing only
        ( '----Z', ( ('CX',(1,4)) , ('CX',(0,4)) ) ) : 'ZZ--Z',  # Lao & Almudever, Fig.1a
        # stabilizers after first 6 CNOTs in Goto Fig.1c
        ( 'Z------', get_sequence('Goto_1c')[:9] ) : 'ZZZ----',
        ( '-Z-----', get_sequence('Goto_1c')[:9] ) : 'XX--X--',
        ( '--Z----', get_sequence('Goto_1c')[:9] ) : 'X-X---X',
        ( '---Z---', get_sequence('Goto_1c')[:9] ) : '---X-XX',
        ( '----Z--', get_sequence('Goto_1c')[:9] ) : '-Z--Z--',
        ( '-----Z-', get_sequence('Goto_1c')[:9] ) : '---Z-Z-',
        ( '------Z', get_sequence('Goto_1c')[:9] ) : '--ZZ--Z',
        
    }
    test_func = lambda input: ''.join(
        clifford_transform_sequence(
            list(input[0]),
            input[1]
            ))
    run_test(test_cases, test_func, 'clifford_transform_sequence')

def test_compute_stabilizer_group():
    """
    Tests the compute_stabilizer_group method.
    """
    test_cases = {
        common_qecc('bitflip_code') + tuple(['XXX']): ('ZZ-', '-ZZ', 'XXX', 'Z-Z', 'YYX', 'YXY', 'XYY')
    }
    test_func = lambda input: tuple([''.join(elem) for elem in compute_stabilizer_group([list(stab) for stab in input])])
    run_test(test_cases, test_func, 'compute_stabilizer_group', outfunc=set)

def test_lowest_weight_equivalent():
    print('>> lowest_weight_equivalent - No Test <<')
    pass

import os
def test_all():
    print(f'\nTesting functions in {os.path.basename(__file__)} ...\n')
    test_compose_paulis()
    test_clifford_transform_dict()
    test_clifford_transform_sequence()
    test_compute_stabilizer_group()
    test_lowest_weight_equivalent()
    print()
    print()


if __name__ == "__main__":
    test_all()





# import numpy as np

# class BSV(object):
#     '''
#     Binary Symplectic Vector for Pauli String
#     '''
#     def __init__(pauli_string='') -> None:
#         Xkeys = {'I':0,'X':1,'Y':1,'Z':0}
#         Zkeys = {'I':0,'X':0,'Y':1,'Z':1}
#         pauli_string = pauli_string
#         x = [Xkeys[p] for p in pauli_string]
#         z = [Zkeys[p] for p in pauli_string]
#         string = '( ' + ' '.join([str(i) for i in x])
#         string += ' | ' + ' '.join([str(i) for i in z]) + ' )'
#         vector = x + z
#         array = np.array(vector)

#     def __repr__():
#         return string

#     def unit_test():
#         test_cases = {
#             'XYIYIIZ': '( 1 1 0 1 0 0 0 | 0 1 0 1 0 0 1 )',
#             }
#         result = [BSV(input).string == output for input, output in test_cases.items()]
#         print(f'BSV - Tests passed: {sum(result)}/{len(result)}')

# BSV().unit_test()
