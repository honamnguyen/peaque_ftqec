import numpy as np
import itertools

def str2tab(pauli_str):
    if type(pauli_str) == list:
        pauli = np.array([list(pstr) for pstr in pauli_str])
    else:
        pauli = pauli_str
    Xkeys = {'I':0,'X':1,'Y':1,'Z':0}
    Zkeys = {'I':0,'X':0,'Y':1,'Z':1}
    return np.hstack([np.vectorize(Zkeys.get)(list(pauli)),
                      np.vectorize(Xkeys.get)(list(pauli))])

def tab2str(pauli_tab):
    keys = {0:'I',1:'Z',-1:'Y',-2:'X'}
    num_qubit = pauli_tab.shape[-1]//2
    result = pauli_tab[...,:num_qubit]-2*pauli_tab[...,num_qubit:]
    try:
        return ''.join(np.vectorize(keys.get)(result))
    except:
        return [''.join(res) for res in np.vectorize(keys.get)(result)]  

def update_gate(indices,Ops):
    if len(indices) == 1:
        update_hadamard(*indices,Ops)
    elif len(indices) == 2:
        update_cnot(*indices,Ops)
    else:
        raise ValueError
    return Ops
def update_cnot(control,target,Ops):
    num_qubit = Ops.shape[-1]//2
    Ops[...,control] = (Ops[...,control] + Ops[...,target]) % 2
    Ops[...,target+num_qubit]  = (Ops[...,control+num_qubit] + Ops[...,target+num_qubit]) % 2
    return Ops

def update_hadamard(ind,Ops):
    num_qubit = Ops.shape[-1]//2
    temp = Ops[...,ind].copy()
    Ops[...,ind] = Ops[...,ind+num_qubit]
    Ops[...,ind+num_qubit] = temp
    return Ops

def fault_operators(num_qubit,indices,fault):
    '''
    Return operators to track in stabilizer simulation for a `fault` happens at location `indices` 
    Must have len(indices) == len(fault)
    e.g. fault_operators(3,[0,1],'YX')
    '''
    Ops = np.zeros([len(indices),2*num_qubit]).astype(int)
    for i,index in enumerate(indices):
        Ops[i,index],Ops[i,num_qubit+index] = str2tab(fault[i])
    return Ops

def stab_equiv(error,stabilizers):
    '''
    Input: a single error string
    Compute all stabilizer equivalent errors, sorted from lowest weight
    '''
    equiv_errors = np.vstack([error,(stabilizers+error)%2])
    return equiv_errors[np.argsort(equiv_errors.sum(1))]

def get_faults(fault_types):
    '''
    Restricted to faults by one- and two-qubit gates
    '''
    one_faults = [''.join(fault) for fault in itertools.product(fault_types,repeat=1)]
    two_faults = [''.join(fault) for fault in itertools.product('I'+fault_types,repeat=2)]
    two_faults.remove('II')
    return [one_faults,two_faults]

def get_flag_error_set(num_qubits,stab_strings,fault_types,gate_seq,verbose=0):
    '''
    Get flag error set from a stabilizer measurement circuit caused by a single fault
    '''
    num_qubit = sum(num_qubits)
    num_data,num_synd,num_flag = num_qubits
    flag_error_set = None
    faults = get_faults(fault_types)
    bin2dec = 2**np.arange(2*num_data)[::-1]

    stabilizers = str2tab(stab_strings)
    # complete the stabilizer group from its generators
    if stabilizers.shape[0]>1:
        extra = []
        for i in range(stabilizers.shape[0]-1):
            for j in range(i+1,stabilizers.shape[0]):
                extra.append((stabilizers[i]+stabilizers[j])%2)
        stabilizers = np.vstack([stabilizers,np.array(extra)])
        
    for i in range(len(gate_seq)):
        # Initialize fault operators after a certain gate
        Ops = []
        fault_set = faults[len(gate_seq[i])-1]
        for fault in fault_set:
            Ops.append(fault_operators(num_qubit,gate_seq[i],fault))
        Ops = np.array(Ops)

        # Run through the remaining circuit
        for gate in gate_seq[i+1:]:
            update_gate(gate,Ops)
        Ops = Ops.sum(-2) % 2
        errors = np.hstack([Ops[:,:num_data],Ops[:,num_qubit:num_qubit+num_data]])
        assert errors.max()<2
        synds = Ops[:,num_qubit+num_data+np.arange(num_synd)] % 2 # X part flips
        flags = Ops[:,num_qubit+num_data+num_synd+np.arange(num_flag)] % 2 # X part flips
        if verbose==2:
            print(f'\n*Fault happening after gate {i+1}*')
            error_strings = tab2str(errors)
            for ii in range(len(fault_set)):
                print(f'\t{fault_set[ii]}:',error_strings[ii],synds[ii],flags[ii])

        # update the flag_error_set when flagged AND encountering a new error
        for flag,error in zip(flags[flags.sum(-1)>0],errors[flags.sum(-1)>0]):
            if flag_error_set is None:
                flag_error_set = np.array([stab_equiv(error,stabilizers)])
                out_flags = [flag]
            elif error@bin2dec not in flag_error_set@bin2dec:
                # print(tab2str(stab_equiv(error,stabilizers)))
                flag_error_set = np.stack([*flag_error_set,stab_equiv(error,stabilizers)])
                out_flags.append(flag)
                
    if verbose:
        print('\n> Flag error set <')
        for flag_error in tab2str(flag_error_set[:,0]):
            print('      ',flag_error)
    
    return np.array(out_flags),flag_error_set[:,0],flag_error_set #reduce the equivalent errors

def check_FT(qec_code,flag_error_set,flags,stab_loc,verbose=0):
    '''
    Input:
        flag_error_set: Set of output errors after a stabilizer circuit when flagged
        stab_loc: Location of the qubits used in a stabilizer circuit w.r.t other qubits in the full code
        
    Output:
        if_fault_tolerant,flag_look_up_table: S2+F1:Error
    '''
    num = len(qec_code[0])
    num_data = len(stab_loc)
    bin2dec = 2**np.arange(len(qec_code)+flags.shape[1])[::-1]
                           
    all_stabilizers = str2tab(qec_code)
    all_stabilizers_twisted = np.hstack([all_stabilizers[:,num:],all_stabilizers[:,:num]])
                         
    error_set = np.zeros([flag_error_set.shape[0],2*num]).astype(int)
    error_set[:,stab_loc]          = flag_error_set[:,:num_data]
    error_set[:,stab_loc+num] = flag_error_set[:,num_data:]
                           
    syndromes = error_set@all_stabilizers_twisted.T %2
    sf = np.hstack([syndromes,flags])@bin2dec # decimal representation of syndrome-flag string
    if verbose==2:
        print('\nSyndromes of flag error set + previous flags:')
        print('S 1 2 3 4 5 6 Flags')
        print('  -----------')
        print(np.hstack([syndromes,flags]))
    if np.unique(sf).size == error_set.shape[0]:
        if verbose: print('--> Fault tolerant!')
        return True,dict(zip(sf,error_set))
    else:
        if verbose: print('--> NOT Fault tolerant!')
        return False,dict(zip(sf,error_set))
    
def lut_decoder(qec_code):
    '''
    LUT decoder for a simple stabilizer distance-3 code
    '''
    num = len(qec_code[0])
    bin2dec = 2**np.arange(len(qec_code))[::-1]
    all_stabilizers = str2tab(qec_code)
    all_stabilizers_twisted = np.hstack([all_stabilizers[:,num:],all_stabilizers[:,:num]])
    
    one_qubit_errors = np.vstack([np.hstack([np.eye(num),np.zeros([num,num])]),
                                  np.hstack([np.zeros([num,num]),np.eye(num)]),
                                  np.hstack([np.eye(num),np.eye(num)])]).astype(int)
    
    syndromes = one_qubit_errors@all_stabilizers_twisted.T %2
    assert np.unique(syndromes@bin2dec).size == one_qubit_errors.shape[0] # unique syndrome for every erro
    return dict(zip(syndromes@bin2dec,one_qubit_errors))