import cirq
import numpy as np
import itertools

def init_qubits(num_qubits):
    num_data,num_syndrome,num_flag = num_qubits
    dat = [cirq.NamedQubit(f'data{i}') for i in range(num_data)]
    synd = [cirq.NamedQubit(f'syndrome{i}') for i in range(num_syndrome)]
    flag = [cirq.NamedQubit(f'flag{i}') for i in range(num_flag)] 

    return np.array([*dat,*synd,*flag])

def create_cirq(q,gate_seq,noise_model=None,readout_noise=None,mmnt_labels=None):
    '''
    New function
    Assume that circuit only has CNOTs and Hadamards
    '''
    circuit = cirq.Circuit(cirq.identity_each(*q))
    imeas = 0
    for loc in gate_seq:
        if 'm' in loc:
            meas_loc = np.array(q)[loc[1:]]
            if readout_noise is not None: 
                circuit += cirq.Moment(readout_noise.on_each(*meas_loc))
            circuit += cirq.measure(*meas_loc,key=mmnt_labels[imeas])
            # circuit += cirq.reset_each(*meas_loc)
    
            imeas += 1
        else:
            if len(loc) == 1:
                op = cirq.H(q[loc[0]])
            else:
                op = cirq.CX(q[loc[0]],q[loc[1]])
            circuit += noise_model.noisy_operation(op) if noise_model is not None else op

    return circuit

# def create_cirq(num_qubits,gate_seq,noise_model=None,readout_noise=None,mmnt_labels=None):
#     '''
#     New function
#     Assume that circuit only has CNOTs and Hadamards
#     '''
    
#     num_data,num_syndrome,num_flag = num_qubits
#     dat = [cirq.NamedQubit(f'data{i}') for i in range(num_data)]
#     synd = [cirq.NamedQubit(f'syndrome{i}') for i in range(num_syndrome)]
#     flag = [cirq.NamedQubit(f'flag{i}') for i in range(num_flag)] 
    
#     circuit = cirq.Circuit()
#     q = [*dat,*synd,*flag]
#     imeas = 0
#     for loc in gate_seq:
#         if 'm' in loc:
#             meas_loc = np.array(q)[loc[1:]]
#             if readout_noise is not None: 
#                 circuit += cirq.Moment(readout_noise.on_each(*meas_loc))
#             circuit += cirq.measure(*meas_loc,key=mmnt_labels[imeas])
#             # circuit += cirq.reset_each(*meas_loc)
    
#             imeas += 1
#         else:
#             if len(loc) == 1:
#                 op = cirq.H(q[loc[0]])
#             else:
#                 op = cirq.CX(q[loc[0]],q[loc[1]])
#             circuit += noise_model.noisy_operation(op) if noise_model is not None else op

#     return circuit,np.array(q)

def create_qsim_circuit(num_qubits,gate_seq,noise_model=None,readout_noise=None,mmnt=None,mmnt_labels=None):
    '''
    Assume that circuit only has CNOTs and Hadamards
    '''
    
    num_data,num_syndrome,num_flag = num_qubits
    dat = [cirq.NamedQubit(f'data{i}') for i in range(num_data)]
    synd = [cirq.NamedQubit(f'syndrome{i}') for i in range(num_syndrome)]
    flag = [cirq.NamedQubit(f'flag{i}') for i in range(num_flag)] 
    
    circuit = cirq.Circuit()
    q = [*dat,*synd,*flag]
    for loc in gate_seq:
        if len(loc) == 1:
            op = cirq.H(q[loc[0]])
        else:
            op = cirq.CX(q[loc[0]],q[loc[1]])
        circuit += noise_model.noisy_operation(op) if noise_model is not None else op
        
    if mmnt is not None:
        parts = [dat,synd,flag]
        for i,measure in enumerate(mmnt):
            if measure:
                if readout_noise is not None: 
                    circuit += cirq.Moment(readout_noise.on_each(*parts[i]))
                circuit += cirq.measure(*parts[i],key=mmnt_labels[i])

    return circuit,q

def signed_probs(mmts,num_qubit):
    '''
    For calculating expectation values of strings of the same pauli
    Eg: <ZZ> = P00 - P01 - P10 + P11
    '''
    outcomes = np.array(list(itertools.product([0,1],repeat=num_qubit)))
    signs = (-1)**(outcomes.sum(1)%2) #minus for odd number of 1s
    probs = []
    for outcome in outcomes:
        probs.append((mmts==outcome).prod(1).sum()/mmts.shape[0])
    return signs*probs

def tab2circ(tab,qubits):
    '''
    Input
        tab: tableau in 1D array
    Output
        action: circuit with corresponding action
    Eg: 3 qubits
        [1,0,1,0,1,1] -> 'ZXY' -> Z(qubit0) + X(qubit1) + Y(qubit2)
    '''
    num = int(len(tab)/2)
    tabZ = tab[:num]
    tabX = tab[num:]
    Z_loc = np.where((tabZ==1) & (tabX==0))
    circ = cirq.Circuit()
    circ += cirq.Z.on_each(qubits[(tabZ==1) & (tabX==0)])
    circ += cirq.X.on_each(qubits[(tabZ==0) & (tabX==1)])
    circ += cirq.Y.on_each(qubits[(tabZ==1) & (tabX==1)])
    return circ