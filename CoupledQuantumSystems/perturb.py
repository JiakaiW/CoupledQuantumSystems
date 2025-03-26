import numpy as np
from abc import ABC, abstractmethod

class Perturbation:
    """
    A base class that encapsulates first- and second-order
    non-degenerate perturbation theory for a given Hamiltonian basis.

    The class stores:
      - dim   : dimension of the Hilbert space
      - E0    : (dim,) unperturbed energies
      - V     : (dim, dim) perturbation operator in the same basis as E0
    and provides methods to compute first- and second-order corrections.
    """

    def __init__(self, dim: int, E0: np.ndarray, V: np.ndarray):
        """
        Args:
            dim : dimension of the Hilbert space
            E0  : shape (dim,), unperturbed energies in the chosen basis
            V   : shape (dim, dim), the perturbation matrix in that same basis
        """
        self.dim = dim
        self.E0 = E0
        self.V = V

    def first_order_perturbation(self):
        # --- (A) First-order energies: E(1) = diag(V) ---
        # Because for the basis state |n>, the first-order shift is V_{n,n}.
        E1 = np.diag(self.V)

        # Summation: E(1st) = E0 + E1
        E1st = self.E0 + E1  # shape (dim,)

        # --- (B) First-order wavefunctions ---
        # In non-degenerate PT, the first-order correction to |n> is:
        #
        #   |ψ_n^(1)> = |n> + sum_{m != n} V_{m,n} / (E0[n] - E0[m]) * |m>
        #
        # We'll build that for all n at once.

        # Build denominator array: denom[m,n] = E0[n] - E0[m]
        E0_col = self.E0.reshape(self.dim, 1)  # shape (dim,1)
        E0_row = self.E0.reshape(1, self.dim)  # shape (1,dim)
        denom = E0_row - E0_col      # shape (dim,dim), [m,n] => E0[n] - E0[m]

        # Avoid dividing by zero on diagonal or near-degenerate
        np.fill_diagonal(denom, np.inf)

        # The coefficient for m != n:
        #   c(m,n) = V[m,n] / (E0[n] - E0[m])
        coeffs = self.V / denom  # shape (dim, dim)

        # The first-order wavefunction for state n is the identity basis vector plus
        # these correction coefficients. So we do:
        psi1st = np.eye(self.dim, dtype=np.complex128) + coeffs

        return E1st, psi1st
    
    def second_order_perturbation(self):
        # (A) First, get the first-order energies & wavefunctions
        E1st, psi1st = self.first_order_perturbation()
        # E1st = E0 + E(1). 
        # psi1st is unnormalized wavefunction up to 1st order.

        # (B) Second-order energy shifts E2[n] = ∑_{m≠n} |V[m,n]|^2 / (E0[n]-E0[m])
        abs_V_sq = np.abs(self.V)**2

        # Denominator array denom[m,n] = E0[n] - E0[m]
        E0_row = self.E0.reshape(1, self.dim)  # shape (1, dim)
        E0_col = self.E0.reshape(self.dim, 1)  # shape (dim, 1)
        denom = E0_row - E0_col      # shape (dim, dim)

        # Avoid dividing by zero on the diagonal
        np.fill_diagonal(denom, np.inf)

        second_order_matrix = abs_V_sq / denom  # shape (dim, dim)

        # sum over m for each column n => E2[n]
        E2 = np.sum(second_order_matrix, axis=0)  # shape (dim,)

        # (C) Second-order wavefunction corrections
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

        numerator = self.V @ psi1st  # shape (dim, dim)
        denom2 = E0_row - E0_col
        np.fill_diagonal(denom2, np.inf)

        psi2 = numerator / denom2  # shape (dim, dim)
        # For safety, zero out the diagonal so we never add infinite self-term
        np.fill_diagonal(psi2, 0.0)

        # (D) The total wavefunction up to second order
        psi_up_to_2 = psi1st + psi2

        # (E) The final energies up to 2nd order:  E(0+1+2) = E1st + E2
        E_up_to_2 = E1st + E2

        for col in range(self.dim):
            norm_col = np.linalg.norm(psi_up_to_2[:, col])
            psi_up_to_2[:, col] /= norm_col

        return E_up_to_2, psi_up_to_2

    @staticmethod
    def operator_in_perturbed_basis(
            psi: np.ndarray,  # (dim, dim), columns = |psi_n^(perturbed)>
            op: np.ndarray            # (dim, dim), operator in the original product basis
        ) -> np.ndarray:
        """
        Compute the matrix elements of `op` in the basis spanned by
        the columns of `psi`. 
        (No normalization or re-orthonormalization is performed.)

        Returns:
            op_dressed: (dim, dim) array = psi^\dagger @ op @ psi
        """
        # Check shapes match
        dim = psi.shape[0]
        assert op.shape == (dim, dim), "Operator must match dimension of psi"
        
        # O_dressed = psi^\dagger @ op @ psi
        op_dressed = psi.conj().T @ op @ psi
        return op_dressed

    @staticmethod
    def reorder_perturbed_states_and_energies(
        energies: np.ndarray,
        states: np.ndarray
    ):
        """
        Sorts energies in ascending order, and reorders the columns of 'states'
        to match that sorting.
        
        Returns:
            E_sorted, states_sorted
        """
        dim = len(energies)
        assert states.shape == (dim, dim), "Dimension mismatch in reorder"

        sort_idx = np.argsort(energies)
        E_sorted = energies[sort_idx]
        states_sorted = states[:, sort_idx]

        return E_sorted, states_sorted

    @staticmethod
    def embed_operator(op: np.ndarray, dims_list: list[int], where_to_embed: int) -> np.ndarray:
        op_list = [np.eye(dim, dtype=op.dtype) for dim in dims_list]
        op_list[where_to_embed] = op
        return np.kron(*op_list)

class TwoBodyPerturbation(Perturbation):
    def __init__(self,         
                 energies_1: np.ndarray,  # shape (m,)
                energies_2: np.ndarray,  # shape (n,)
                n_op_1: np.ndarray,      # shape (m,m)
                n_op_2: np.ndarray,      # shape (n,n)
                g: float
            ):
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
        super().__init__(dim, E0, V)

    def two_body_second_order_perturbation_vectorized(self) -> tuple[np.ndarray, np.ndarray]:
        return self.second_order_perturbation()

    def two_body_first_order_perturbation_vectorized(self) -> tuple[np.ndarray, np.ndarray]:
        return self.first_order_perturbation()