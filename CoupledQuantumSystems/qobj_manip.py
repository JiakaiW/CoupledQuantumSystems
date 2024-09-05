import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import scqubits
import qutip
from typing import Union


############################################################################
#
#
# Functions about manipulating qutip/scqubit objects
#
#
############################################################################



def truncate_custom(qobj: qutip.Qobj, products_to_keep: list, product_to_dressed: dict) -> qutip.Qobj:
    products_to_keep_tuples = [tuple(product) for product in products_to_keep]
    
    # Find indices to keep based on matching products_to_keep with keys in product_to_dressed
    indices_to_keep = [dressed_level for product, dressed_level in product_to_dressed.items() 
                       if product in products_to_keep_tuples]
    try:
        indices_to_keep.sort()
    except:
        print(indices_to_keep)

    if qobj.isket:  # is ket
        truncated_vector = qobj.full()[indices_to_keep, :]
        return qutip.Qobj(truncated_vector)
    elif qobj.isoper or qobj.isoperket:
        truncated_matrix = qobj.full()[np.ix_(indices_to_keep, indices_to_keep)]
        return qutip.Qobj(truncated_matrix)
    elif qobj.issuper:
        data = qobj.full()
        data = data.reshape((qobj.dims[0][0][0],qobj.dims[0][1][0],qobj.dims[0][0][0],qobj.dims[0][1][0]))
        data = data[np.ix_(indices_to_keep,indices_to_keep,indices_to_keep, indices_to_keep)]
        data = data.reshape((len(products_to_keep)**2, len(products_to_keep)**2))
        return qutip.Qobj(data)
    else:
        raise ValueError("Unsupported qobj type. Please provide a ket, operator, or superoperator.")
    
def pad_back_custom(qobj: qutip.Qobj, products_to_keep: Union[list,None], product_to_dressed: dict) -> qutip.Qobj:
    if products_to_keep == None:
        # for compatibility
        return qobj
    products_to_keep_tuples = [tuple(product) for product in products_to_keep]
    indices_to_keep = [dressed_level for product, dressed_level in product_to_dressed.items() 
                       if product in products_to_keep_tuples]
    indices_to_keep.sort()

    full_dimension = max(product_to_dressed.values()) + 1

    if qobj.shape[1] == 1:  # is ket
        padded_vector = np.zeros((full_dimension, 1), dtype=complex)
        padded_vector[indices_to_keep, :] = qobj.full()
        return qutip.Qobj(padded_vector)
    else:  # is operator or density matrix
        padded_matrix = np.zeros((full_dimension, full_dimension), dtype=complex)
        padded_matrix[np.ix_(indices_to_keep, indices_to_keep)] = qobj.full()
        return qutip.Qobj(padded_matrix)


def generate_single_mapping(H_with_interaction_no_drive,evals = None, evecs = None) -> np.ndarray:
    """
    The input should be in product basis
    Maps product of bare states to dressed state
    Returns a dictionary like {(0,0,0):0,(0,0,1):1}
    Use this function instead of scqubit's because I can change the overlap threshold here
    """
    if evals is None or evecs is None:
        evals, evecs = H_with_interaction_no_drive.eigenstates()
    overlap_matrix = scqubits.utils.spectrum_utils.convert_evecs_to_ndarray(evecs)
    OVERLAP_THRESHOLD = 0.1
    dims = H_with_interaction_no_drive.dims[0]
    system_size = len(dims)

    product_state_names = list(itertools.product(*[range(dim) for dim in dims]))

    total_dim = math.prod(dims)
    dressed_indices_of_product_states = [None] * total_dim

    # for every energy eigenstate, from the lowerst to the highest, find the product state
    for dressed_index in range(len(evals)):
        max_position = (np.abs(overlap_matrix[dressed_index, :])).argmax()
        max_overlap = np.abs(overlap_matrix[dressed_index, max_position])
        overlap_matrix[:, max_position] = 0
        dressed_indices_of_product_states[int(max_position)] = dressed_index
        if (max_overlap**2 < OVERLAP_THRESHOLD):
            print(f'max overlap^2 {max_overlap**2} below threshold for dressed state {dressed_index} with eval {evals[dressed_index]}')
    product_to_dressed = {}
    for product, dressed in zip(product_state_names,dressed_indices_of_product_states):
        product_to_dressed[product] = dressed
    return product_to_dressed


def dressed_to_2_level_dm(dressed_dm: qutip.Qobj, 
                        product_to_dressed: dict, 
                        qbt_position: int,
                        filtered_product_to_dressed: dict,
                        sign_multiplier:dict,
                        products_to_keep=None,
                        ) -> qutip.Qobj:
    """
    Convert a dressed density matrix to a multi-level density matrix for specified computational states,
    inferring subsystem dimensions from product_to_dressed.

    Parameters:
    - dressed_dm: The dressed density matrix as a qutip.Qobj.
    - product_to_dressed: Mapping from product states to dressed states indices.
    - qbt_position: which of the subsystem is the qubit
    - filtered_product_to_dressed:levels relevent to the qubit computational states
    - sign_multiplier: 
    - products_to_keep: Optional list of product states to keep.

    Returns:
    - qutip.Qobj representing the reduced density matrix for specified computational states.
    """
    dressed_dm_data = pad_back_custom(dressed_dm, products_to_keep, product_to_dressed)
    if dressed_dm_data.shape[1] == 1:
        dressed_dm_data = qutip.ket2dm(dressed_dm_data)
    dressed_dm_data = dressed_dm_data.full()

    # Infer subsystem dimensions
    subsystem_dims = [max(indexes) + 1 for indexes in zip(*product_to_dressed.keys())]
    subsystem_dims[qbt_position] = 2
    rho_product = np.zeros((subsystem_dims*2), dtype=complex) # Here rho_product is shaped like (dim1,dim2,dim1,dim2)
    for product_state, dressed_index1 in filtered_product_to_dressed.items():
        for product_state2, dressed_index2 in filtered_product_to_dressed.items():
            element = dressed_dm_data[dressed_index1, dressed_index2] * sign_multiplier[dressed_index1] * sign_multiplier[dressed_index2]
            rho_product[product_state+product_state2] += element # Using index like (lvl1, lvl2, lvl1, lvl2) to access of of the entries

    two_lvl_qbt_dm_size = np.prod(subsystem_dims)
    rho_product = rho_product.reshape((two_lvl_qbt_dm_size,two_lvl_qbt_dm_size))
    rho_product = qutip.Qobj(rho_product, dims=[subsystem_dims, subsystem_dims])

    rho_2_level = rho_product.ptrace(qbt_position)
    return rho_2_level


def find_dominant_frequency(expectation,tlist,dominant_frequency_already_found = None,plot = False,plot_freq = False):
    # In case alpha oscillates not at drive frequency, we do fourier transform to make the plot of coherent state look better 

    if dominant_frequency_already_found != None:
        expectation = expectation * np.exp(-1j*2*np.pi*dominant_frequency_already_found*tlist)

    expectation_fft = np.fft.fft(expectation)
    frequencies = np.fft.fftfreq(len(tlist), d=(tlist[1] - tlist[0]))  # assuming tlist is uniformly spaced

    # Identify the dominant frequency: 
    # (we exclude the zero frequency, which usually has the DC offset)
    dominant_freq_idx = np.argmax(np.abs(expectation_fft[1:])) + 1
    dominant_freq = frequencies[dominant_freq_idx]

    if plot:
        # Print the dominant frequency
        print(f"The dominant oscillation frequency is: {dominant_freq:.3f} (in the same units as 1/timestep)")

        fft_shifted = np.fft.fftshift(expectation_fft)
        frequencies_shifted = np.fft.fftshift(frequencies)
        plt.plot(frequencies_shifted, np.abs(fft_shifted))
        plt.xlabel('Frequency (arbitrary units)')
        plt.ylabel('Magnitude')
        plt.title('FFT of the Expectation Value')
        plt.grid(True)
        plt.show()
    elif plot_freq:
        plt.plot(expectation_fft)
        plt.show()
    else:
        return dominant_freq


def transition_frequency(hilbertspace,s0: int, s1: int) -> float:
    return (hilbertspace.energy_by_dressed_index(s1)- hilbertspace.energy_by_dressed_index(s0))


# Deprecated
def get_product(dressed_dm,pad_back_custom,product_to_dressed,sign_multiplier):
    '''
    This function converts a dressed density matrix to product basis, taking into consideration the sign difference between dressed basis and product basis.
    Here, "sign difference" means that, for almost all pacakges (scipy, numpy, jax.numpy), when computing the eigensystem, there are eigenvectors that's shaped like [0,0,-1,0,0]. 
        but when I use eigenstates as initial states, I use [0,0,1,0,0]. 
    '''
    dressed_dm_data = pad_back_custom(dressed_dm)
    # if dressed_dm_data.shape[1] == 1:
    #     dressed_dm_data = qutip.ket2dm(dressed_dm_data)
    dressed_dm_data = dressed_dm_data.full() # 2d np.array

    # Infer subsystem dimensions
    subsystem_dims = [max(indexes) + 1 for indexes in zip(*product_to_dressed.keys())]
    rho_product = np.zeros((subsystem_dims*2), dtype=complex) # Here rho_product is shaped like (dim1,dim2,dim1,dim2)

    # unvectorized
    for product_state, dressed_index1 in product_to_dressed.items():
        for product_state2, dressed_index2 in product_to_dressed.items():
            element = dressed_dm_data[dressed_index1, dressed_index2] * sign_multiplier[dressed_index1] * sign_multiplier[dressed_index2]
            rho_product[product_state+product_state2] += element # Using index like (lvl1, lvl2, lvl1, lvl2) to access of of the entries

    two_lvl_qbt_dm_size = np.prod(subsystem_dims)
    rho_product = rho_product.reshape((two_lvl_qbt_dm_size,two_lvl_qbt_dm_size))
    rho_product = qutip.Qobj(rho_product, dims=[subsystem_dims, subsystem_dims])
    return rho_product

def get_product_vectorized(dressed_dm,pad_back_custom,product_to_dressed,sign_multiplier):
    '''
    This function converts a dressed density matrix to product basis, taking into consideration the sign difference between dressed basis and product basis.
    Here, "sign difference" means that, for almost all pacakges (scipy, numpy, jax.numpy), when computing the eigensystem, there are eigenvectors that's shaped like [0,0,-1,0,0]. 
        but when I use eigenstates as initial states, I use [0,0,1,0,0]. 
    '''
    dressed_dm_data = pad_back_custom(dressed_dm)
    # if dressed_dm_data.shape[1] == 1:
    #     dressed_dm_data = qutip.ket2dm(dressed_dm_data)
    dressed_dm_data = dressed_dm_data.full() # 2d np.array

    subsystem_dims = [max(indexes) + 1 for indexes in zip(*product_to_dressed.keys())]
    rho_product =dressed_to_product_vectorized(product_to_dressed, dressed_dm_data, sign_multiplier,subsystem_dims) 

    two_lvl_qbt_dm_size = np.prod(subsystem_dims)
    rho_product = rho_product.reshape((two_lvl_qbt_dm_size,two_lvl_qbt_dm_size))
    rho_product = qutip.Qobj(rho_product, dims=[subsystem_dims, subsystem_dims])
    return rho_product

def dressed_to_product_vectorized(product_to_dressed, dressed_dm_data, sign_multiplier,subsystem_dims=None):
    if subsystem_dims is None:
        # Step 1: Get the subsystem dimensions (generalized for any number of dimensions)
        subsystem_dims = [max(indexes) + 1 for indexes in zip(*product_to_dressed.keys())]
    num_dims = len(subsystem_dims)
    
    # Step 2: Construct the shape for rho_product (twice the number of dimensions)
    rho_product = np.zeros(subsystem_dims * 2, dtype=complex)

    # Step 3: Extract product states and dressed indices
    product_states = np.array(list(product_to_dressed.keys()))  # Shape: (num_states, num_dimensions)
    dressed_indices = np.array(list(product_to_dressed.values()))  # Shape: (num_states,)
    
    # Step 4: Get the dressed submatrix and sign multipliers
    dressed_dm_submatrix = dressed_dm_data[np.ix_(dressed_indices, dressed_indices)]  # Shape: (num_states, num_states)
    sign_multipliers = sign_multiplier[dressed_indices]  # Shape: (num_states,)
    
    # Step 5: Apply sign multiplier to the dressed_dm_submatrix
    element_matrix = dressed_dm_submatrix * np.outer(sign_multipliers, sign_multipliers)  # Shape: (num_states, num_states)

    # Step 6: Prepare indices for updating rho_product using numpy
    rho_indices1 = []
    rho_indices2 = []
    # For each dimension of the product state, we create the corresponding indices for the first and second parts
    for dim in range(num_dims):
        idx1, idx2 = np.meshgrid(product_states[:, dim], product_states[:, dim], indexing='ij')  # First and second indices for this dimension
        rho_indices1.append(idx1)
        rho_indices2.append(idx2)

    # Step 7: Convert list of indices into a tuple for advanced indexing
    rho_indices = tuple(rho_indices1+rho_indices2)  # This will now be a tuple of length `2 * num_dims`

    # Step 8: Use advanced indexing to update rho_product
    rho_product[rho_indices] += element_matrix

    return rho_product