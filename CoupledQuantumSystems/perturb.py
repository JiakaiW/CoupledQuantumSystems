import numpy as np
import numpy as np

def first_order_perturbation_vectorized(
    energies_1: np.ndarray,  # shape (m,)
    energies_2: np.ndarray,  # shape (n,)
    n_op_1: np.ndarray,      # shape (m, m)
    n_op_2: np.ndarray,      # shape (n, n)
    g: float
):
    """
    Compute first-order perturbation theory corrections for two multi-level qubits
    coupled by V = g * n_1 * n_2, using a fully vectorized approach.

    Returns:
        corrected_energies:    1D array of length m*n with the first-order
                               corrected energies in product-basis order.
        corrected_vectors:     (m*n, m*n) array whose columns are the
                               first-order-corrected eigenvectors in the
                               same product-basis ordering.
    """

    m = energies_1.size
    n = energies_2.size

    # -- 1) Unperturbed 2-qubit energies in a shape-(m,n) array
    E0 = energies_1[:, None] + energies_2[None, :]  # shape (m,n)

    # -- 2) First-order energy shifts
    diag1 = np.diag(n_op_1)  # shape (m,)
    diag2 = np.diag(n_op_2)  # shape (n,)
    E1 = g * diag1[:, None] * diag2[None, :]        # shape (m,n)
    corrected_energies = (E0 + E1).ravel()          # shape (m*n,)

    # -- 3) Compute first-order corrections to eigenvectors

    # 3a) The unperturbed wavefunction is effectively the identity in the product basis
    #     (each basis vector is "1" along its own axis, "0" elsewhere),
    #     so start with an identity matrix:
    m_tot = m * n
    corrected_vectors = np.eye(m_tot, dtype=complex)  # shape (m*n, m*n)

    # 3b) Build the matrix elements of the perturbation V = g n_op_1 n_op_2
    #     in a shape-(m, n, m, n) array called mat_elem[k, l, i, j]
    #     = g * n_op_1[k,i] * n_op_2[l,j].
    #
    #     We can do this neatly with `np.einsum` or by broadcasting:
    mat_elem = np.einsum("ki,lj->kilj", n_op_1, n_op_2) * g  
    # mat_elem now has shape (m, m, n, n), with indices (k, i, l, j).
    # We want it as (k, l, i, j), so just reorder axes:
    mat_elem = mat_elem.transpose(0, 2, 1, 3)  # shape = (m, n, m, n)
    # where the axes are [k, l, i, j].

    # 3c) Build the denominator E0(i,j) - E0(k,l) in shape (m, n, m, n).
    #     We want denom[k, l, i, j] = E0[i, j] - E0[k, l].
    E0_kl = E0[:, :, None, None]     # shape (m, n, 1, 1)
    E0_ij = E0[None, None, :, :]     # shape (1, 1, m, n)
    denom = E0_ij - E0_kl            # shape (m, n, m, n) broadcasting

    # 3d) First-order correction amplitudes:
    #     c_{kl,ij} = mat_elem[k,l,i,j] / [E0(i,j) - E0(k,l)] for (k,l) != (i,j).
    c = np.zeros_like(mat_elem, dtype=complex)  # shape (m, n, m, n)
    # Avoid division by zero for degenerate or same states:
    small_mask = (np.abs(denom) < 1e-14)
    # We skip these terms in non-degenerate first-order PT:
    valid_mask = ~small_mask

    # Evaluate c where valid:
    c[valid_mask] = mat_elem[valid_mask] / denom[valid_mask]

    # Zero out the self term explicitly (k,l)==(i,j):
    #   because standard 1st-order formula is sum over (k,l) != (i,j).
    #   That means c[k,l,i,j] = 0 if k==i and l==j.
    k_index, l_index, i_index, j_index = np.indices((m, n, m, n))
    same_state_mask = (k_index == i_index) & (l_index == j_index)
    c[same_state_mask] = 0.0

    # 3e) Insert these amplitudes into the columns of `corrected_vectors`.
    #     We want:
    #       row = flattenIndex(k,l) = l*m + k,
    #       col = flattenIndex(i,j) = j*m + i,
    #       corrected_vectors[row, col] = c_{k,l,i,j}.
    #
    #     We'll create rowIndex and colIndex arrays of shape (m,n,m,n),
    #     then flatten them along with c to do a single advanced assignment.
    #
    def flatten_index_2d(i2d, j2d, dim_size_m):
        """Convert (i2d, j2d) -> j2d*dim_size_m + i2d elementwise."""
        return j2d * dim_size_m + i2d

    # shape (m,n,m,n) for each index array
    row_index_4d = flatten_index_2d(k_index, l_index, m)  # shape (m,n,m,n)
    col_index_4d = flatten_index_2d(i_index, j_index, m)  # shape (m,n,m,n)

    # Flatten all of them consistently:
    row_flat = row_index_4d.ravel()
    col_flat = col_index_4d.ravel()
    c_flat   = c.ravel()

    # Add to the identity (which was the unperturbed wavefunction)
    # So the final amplitude is 1 on (i,j) plus the sum of all c_{k,l} offsets.
    # We'll do += because the identity is already in corrected_vectors.
    corrected_vectors[row_flat, col_flat] += c_flat

    return corrected_energies, corrected_vectors


def first_order_perturbation(
    energies_1: np.ndarray,  # array of length m (unperturbed energies of qubit 1)
    energies_2: np.ndarray,  # array of length n (unperturbed energies of qubit 2)
    n_op_1: np.ndarray,      # m x m charge operator for qubit 1 in the same basis
    n_op_2: np.ndarray,      # n x n charge operator for qubit 2 in the same basis
    g: float                 # coupling strength
):
    """
    Returns:
        corrected_energies:   (m*n,) array with first-order shifted energies in product basis ordering
        corrected_eigenvectors: (m*n, m*n) array whose columns are the first-order corrected eigenvectors 
                                in the same product basis ordering.
                                
    The product basis ordering used is (i, j) -> j*m + i, so index = j*m + i.
    
    Assumes non-degenerate first-order perturbation theory. Degeneracies must
    be treated more carefully.
    """
    
    m = len(energies_1)
    n = len(energies_2)
    
    # Allocate space for results
    corrected_energies = np.zeros(m*n, dtype=np.float64)
    # We'll keep the vectors as complex to be safe, in case n_op_1 or n_op_2 have off-diagonal phases
    corrected_eigenvectors = np.zeros((m*n, m*n), dtype=np.complex128)
    
    # Loop over each unperturbed product state |i> x |j>
    for i in range(m):
        for j in range(n):
            idx = j*m + i  # Flattened index in the product basis
            
            # Unperturbed energy
            E0 = energies_1[i] + energies_2[j]
            
            # First-order energy shift = g * <i|n_op_1|i> * <j|n_op_2|j>
            # (since the unperturbed state is a product state in this basis)
            E1 = g * n_op_1[i, i] * n_op_2[j, j]
            corrected_energies[idx] = E0 + E1
            
            # Construct the first-order corrected wavefunction
            # Start with the unperturbed component (the basis vector e_{ij})
            vec = np.zeros(m*n, dtype=np.complex128)
            vec[idx] = 1.0  # The zeroth-order wavefunction
            
            # Add the sum over all (k,l) != (i,j)
            for k in range(m):
                for l in range(n):
                    if (k != i) or (l != j):
                        E_kl = energies_1[k] + energies_2[l]
                        denom = E0 - E_kl
                        
                        # If there's exact degeneracy (denom ~ 0), the standard
                        # non-degenerate formula breaks down; we skip or handle carefully.
                        if abs(denom) < 1e-14:
                            continue
                            
                        # Matrix element <k,l|g n_1 n_2|i,j> = g * <k|n_1|i> <l|n_2|j>
                        mat_elem = g * n_op_1[k, i] * n_op_2[l, j]
                        c_kl = mat_elem / denom
                        
                        # Add that component to the wavefunction
                        kl_index = l*m + k
                        vec[kl_index] += c_kl
            
            # Place this vector into the eigenvector array as a column
            corrected_eigenvectors[:, idx] = vec
    
    return corrected_energies, corrected_eigenvectors
