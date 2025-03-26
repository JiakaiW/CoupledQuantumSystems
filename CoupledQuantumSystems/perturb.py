import numpy as np


def first_order_perturbation_vectorized(
    energies_1: np.ndarray,   # shape (m,)
    energies_2: np.ndarray,   # shape (n,)
    n_op_1: np.ndarray,       # shape (m, m)
    n_op_2: np.ndarray,       # shape (n, n)
    g: float
):
    m = len(energies_1)
    n = len(energies_2)
    dim = m * n

    # --- First-order corrected energies ---
    E1_grid, E2_grid = np.meshgrid(energies_2, energies_1)  # shape (m, n)
    E0 = E1_grid + E2_grid  # shape (m, n)

    n1_diag = np.diag(n_op_1)  # shape (m,)
    n2_diag = np.diag(n_op_2)  # shape (n,)
    n1n2_grid = np.outer(n1_diag, n2_diag)  # shape (m, n)

    corrected_energies = E0 + g * n1n2_grid  # shape (m, n)
    corrected_energies = corrected_energies.ravel(order="C")  # shape (m*n,)

    # --- First-order corrected eigenvectors ---
    # Unperturbed energy grid for all (i,j), (k,l) pairs
    E0_ij = E0.ravel(order="C")[:, None]  # shape (mn, 1)
    E0_kl = E0.ravel(order="C")[None, :]  # shape (1, mn)
    denom = E0_ij - E0_kl  # shape (mn, mn)

    # Avoid division by zero (degeneracies)
    np.fill_diagonal(denom, np.inf)

    # Compute matrix elements: <k|n1|i> * <l|n2|j>
    M1 = n_op_1  # shape (m, m)
    M2 = n_op_2  # shape (n, n)
    mat_elem = g * np.kron(M1, M2)  # shape (mn, mn)

    coeffs = mat_elem / denom  # shape (mn, mn)

    # Add identity (zeroth order term)
    corrected_eigenvectors = np.eye(dim, dtype=np.complex128) + coeffs

    return corrected_energies, corrected_eigenvectors



def first_order_perturbation(
    energies_1: np.ndarray,  # array of length m (qubit 1 energies)
    energies_2: np.ndarray,  # array of length n (qubit 2 energies)
    n_op_1: np.ndarray,      # m x m charge operator for qubit 1
    n_op_2: np.ndarray,      # n x n charge operator for qubit 2
    g: float                 # interaction strength
):
    """
    Approximates eigenvalues and eigenvectors of a two-qubit system with interaction 
    g * n1 ⊗ n2 using first-order perturbation theory.

    Product basis is ordered as (i, j) -> index = i * n + j.

    Returns:
        corrected_energies: 1D array of length m * n with corrected energies
        corrected_eigenvectors: 2D array (m*n, m*n) where each column is a corrected eigenvector
    """
    m = len(energies_1)
    n = len(energies_2)
    
    corrected_energies = np.zeros(m * n, dtype=np.float64)
    corrected_eigenvectors = np.zeros((m * n, m * n), dtype=np.complex128)

    for i in range(m):
        for j in range(n):
            idx = i * n + j  # Your product basis ordering

            E0 = energies_1[i] + energies_2[j]
            E1 = g * n_op_1[i, i] * n_op_2[j, j]
            corrected_energies[idx] = E0 + E1

            # Start with unperturbed eigenvector
            vec = np.zeros(m * n, dtype=np.complex128)
            vec[idx] = 1.0

            for k in range(m):
                for l in range(n):
                    if (k == i) and (l == j):
                        continue

                    denom = E0 - (energies_1[k] + energies_2[l])
                    if abs(denom) < 1e-14:
                        continue  # Degeneracy: skip or treat specially

                    mat_elem = g * n_op_1[k, i] * n_op_2[l, j]
                    coeff = mat_elem / denom
                    kl_idx = k * n + l  # Updated product basis index
                    vec[kl_idx] += coeff

            corrected_eigenvectors[:, idx] = vec

    return corrected_energies, corrected_eigenvectors
