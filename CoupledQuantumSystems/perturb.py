import numpy as np

def second_order_perturbation_vectorized(
    energies_1: np.ndarray,  # shape (m,)
    energies_2: np.ndarray,  # shape (n,)
    n_op_1: np.ndarray,      # shape (m,m)
    n_op_2: np.ndarray,      # shape (n,n)
    g: float
):
    """
    Compute two-qubit energies and wavefunctions up to second-order PT,
    reusing the first-order function.

    Returns:
       E0          : (dim,) unperturbed energies
       E1st        : (dim,) energies up to first order = E0 + E(1)
       E2          : (dim,) the second-order energy shifts
       E_up_to_2   : (dim,) the final energies E(0+1+2)
       
       psi1st      : (dim, dim) wavefunctions up to first order
                     columns are states. 
       psi2        : (dim, dim) the second-order *correction* wavefunctions
       psi_up_to_2 : (dim, dim) wavefunctions up to second order 
                     = psi1st + psi2
    """

    # (A) First, get the first-order energies & wavefunctions
    E1st, psi1st = first_order_perturbation_vectorized(
        energies_1, energies_2, n_op_1, n_op_2, g
    )
    # E1st = E0 + E(1). 
    # psi1st is unnormalized wavefunction up to 1st order.

    m = len(energies_1)
    n = len(energies_2)
    dim = m * n

    # (B) Build unperturbed energies E0 again (needed for 2nd order denominators)
    E2_grid, E1_grid = np.meshgrid(energies_2, energies_1)  # shape (m,n)
    E0_2D = E1_grid + E2_grid
    E0 = E0_2D.ravel(order="C")  # shape (dim,)

    # (C) Build the perturbation matrix V again for second-order sums
    V = g * np.kron(n_op_1, n_op_2)  # shape (dim, dim)

    # (D) Second-order energy shifts E2[n] = ∑_{m≠n} |V[m,n]|^2 / (E0[n]-E0[m])
    abs_V_sq = np.abs(V)**2

    # Denominator array denom[m,n] = E0[n] - E0[m]
    E0_row = E0.reshape(1, dim)  # shape (1, dim)
    E0_col = E0.reshape(dim, 1)  # shape (dim, 1)
    denom = E0_row - E0_col      # shape (dim, dim)

    # Avoid dividing by zero on the diagonal
    np.fill_diagonal(denom, np.inf)

    second_order_matrix = abs_V_sq / denom  # shape (dim, dim)

    # sum over m for each column n => E2[n]
    E2 = np.sum(second_order_matrix, axis=0)  # shape (dim,)

    # (E) Second-order wavefunction corrections
    # 
    #   |psi_n^(2)> = ∑_{m ≠ n}  (⟨m|V|psi_n^(1)> / [E0[n]-E0[m]])  |m>
    #
    # We'll do this in a vectorized way by:
    #   numerator = V @ psi1st   (matrix-matrix product, shape (dim, dim))
    # so column n is V * (psi_n^(1))
    #   => numerator[m,n] = ∑_p V[m,p]*psi1st[p,n] = ⟨m|V|psi_n^(1)⟩
    #
    # Then denom2[m,n] = E0[n] - E0[m], same shape (dim, dim).
    # => c2[m,n] = numerator[m,n] / denom2[m,n] for m != n.

    numerator = V @ psi1st  # shape (dim, dim)
    denom2 = E0_row - E0_col
    np.fill_diagonal(denom2, np.inf)

    psi2 = numerator / denom2  # shape (dim, dim)
    # For safety, zero out the diagonal so we never add infinite self-term
    np.fill_diagonal(psi2, 0.0)

    # (F) The total wavefunction up to second order
    psi_up_to_2 = psi1st + psi2

    # (G) The final energies up to 2nd order:  E(0+1+2) = E1st + E2
    E_up_to_2 = E1st + E2

    for col in range(dim):
        norm_col = np.linalg.norm(psi_up_to_2[:, col])
        psi_up_to_2[:, col] /= norm_col

    return E_up_to_2, psi_up_to_2




def first_order_perturbation_vectorized(
    energies_1: np.ndarray,   # shape (m,)
    energies_2: np.ndarray,   # shape (n,)
    n_op_1: np.ndarray,       # shape (m, m)
    n_op_2: np.ndarray,       # shape (n, n)
    g: float
):
    """
    Return:
        E1st    : (m*n,) array of energies up to first order
                  (i.e. unperturbed E0 + first-order corrections).
        psi1st  : (m*n, m*n) array, whose columns are the unnormalized
                  wavefunctions up to first order for each product state.
                  
    Product basis ordering: (i, j) -> index = i*n + j.
    """
    m = len(energies_1)
    n = len(energies_2)
    dim = m * n

    # --- (A) Unperturbed energies E0 in product basis ---
    # E0_ij = E_i^(1) + E_j^(2)
    E1_grid, E2_grid = np.meshgrid(energies_2, energies_1)  # shape = (m, n)
    E0_2D = E1_grid + E2_grid
    E0 = E0_2D.ravel(order="C")  # shape (m*n,)

    # --- (B) Full perturbation matrix V = g * n_1 ⊗ n_2 ---
    V = g * np.kron(n_op_1, n_op_2)  # shape (dim, dim)

    # --- (C) First-order energies: E(1) = diag(V) ---
    # Because for the basis state |n>, the first-order shift is V_{n,n}.
    E1 = np.diag(V)

    # Summation: E(1st) = E0 + E1
    E1st = E0 + E1  # shape (dim,)

    # --- (D) First-order wavefunctions ---
    # In non-degenerate PT, the first-order correction to |n> is:
    #
    #   |ψ_n^(1)> = |n> + sum_{m != n} V_{m,n} / (E0[n] - E0[m]) * |m>
    #
    # We'll build that for all n at once.

    # Build denominator array: denom[m,n] = E0[n] - E0[m]
    E0_col = E0.reshape(dim, 1)  # shape (dim,1)
    E0_row = E0.reshape(1, dim)  # shape (1,dim)
    denom = E0_row - E0_col      # shape (dim,dim), [m,n] => E0[n] - E0[m]

    # Avoid dividing by zero on diagonal or near-degenerate
    np.fill_diagonal(denom, np.inf)

    # The coefficient for m != n:
    #   c(m,n) = V[m,n] / (E0[n] - E0[m])
    coeffs = V / denom  # shape (dim, dim)

    # The first-order wavefunction for state n is the identity basis vector plus
    # these correction coefficients. So we do:
    psi1st = np.eye(dim, dtype=np.complex128) + coeffs

    return E1st, psi1st

def first_order_perturbation_vectorized_old(
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
