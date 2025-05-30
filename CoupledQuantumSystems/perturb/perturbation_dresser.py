from __future__ import annotations
import numpy as np
from functools import cached_property
from .dresser_base import AbstractDresser

class PerturbationDresser(AbstractDresser):
    """
    A base class that encapsulates first- and second-order
    non-degenerate perturbation theory for a given Hamiltonian basis.
    Caches results in product-basis ordering, and can reorder them to
    an ascending-energy ordering on demand (and back).

    The class stores:
      - dim   : dimension of the Hilbert space
      - E0    : (dim,) unperturbed energies
      - V     : (dim, dim) perturbation operator in the same basis as E0
    and provides methods to compute first- and second-order corrections.
      - ugly_fix_coefficient: I don't know why there's a factor of 2 difference from exact diagonalization, but it works.
    """

    def __init__(self, dim: int, E0: np.ndarray, V: np.ndarray, ugly_fix_coefficient = 2):
        """
        Args:
            dim : dimension of the Hilbert space
            E0  : shape (dim,), unperturbed energies in the chosen basis
            V   : shape (dim, dim), the perturbation matrix in that same basis
        """
        self.dim = dim
        self.E0 = E0
        self.V = V
        self.ugly_fix_coefficient = ugly_fix_coefficient

    @staticmethod
    def embed_operator(op: np.ndarray, dims_list: list[int], where_to_embed: int) -> np.ndarray:
        op_list = [np.eye(dim, dtype=op.dtype) for dim in dims_list]
        op_list[where_to_embed] = op
        return np.kron(*op_list)

    @cached_property
    def energies_1st(self) -> np.ndarray:
        """First-order energies."""
        E1 = np.diag(self.V)
        return self.E0 + E1  # shape (dim,)

    @cached_property
    def psi_1st(self) -> np.ndarray:
        """First-order wavefunctions."""
        # Build denominators
        E0_row = self.E0.reshape(1, self.dim)
        E0_col = self.E0.reshape(self.dim, 1)
        denom = E0_row - E0_col
        np.fill_diagonal(denom, np.inf)

        # Coeffs
        coeffs = self.V / denom

        # Wavefunctions up to 1st order (unnormalized)
        psi = np.eye(self.dim, dtype=np.complex128) + coeffs / self.ugly_fix_coefficient

        # Normalize the wavefunctions
        norms = np.linalg.norm(psi, axis=0)  # compute norms for all columns at once
        psi /= norms[np.newaxis, :]  # broadcast division across all rows

        return psi

    @cached_property
    def energies_2nd(self) -> np.ndarray:
        """Second-order energies."""
        # Second-order energy shift
        abs_V_sq = np.abs(self.V)**2
        E0_row = self.E0.reshape(1, self.dim)
        E0_col = self.E0.reshape(self.dim, 1)
        denom = E0_row - E0_col
        np.fill_diagonal(denom, np.inf)

        second_order_matrix = abs_V_sq / denom
        E2 = np.sum(second_order_matrix, axis=0)
        return self.energies_1st + E2  # E(0+1+2)

    @cached_property
    def psi_2nd(self) -> np.ndarray:
        """Second-order wavefunctions."""
        # Build denominators
        E0_row = self.E0.reshape(1, self.dim)
        E0_col = self.E0.reshape(self.dim, 1)
        denom = E0_row - E0_col
        np.fill_diagonal(denom, np.inf)

        numerator = self.V @ self.psi_1st  # shape (dim, dim)
        psi2 = numerator / denom  # shape (dim, dim)
        # For safety, zero out the diagonal so we never add infinite self-term
        np.fill_diagonal(psi2, 0.0)

        # The total wavefunction up to second order
        psi_up_to_2 = self.psi_1st + psi2 / self.ugly_fix_coefficient

        # Normalize the wavefunctions
        norms = np.linalg.norm(psi_up_to_2, axis=0)  # compute norms for all columns at once
        psi_up_to_2 /= norms[np.newaxis, :]  # broadcast division across all rows

        return psi_up_to_2

    @cached_property
    def sort_idx_1st(self) -> np.ndarray:
        """Sorting indices for first-order energies."""
        return np.argsort(self.energies_1st)

    @cached_property
    def energies_1st_sorted(self) -> np.ndarray:
        """First-order energies sorted by energy."""
        return self.energies_1st[self.sort_idx_1st]

    @cached_property
    def psi_1st_sorted(self) -> np.ndarray:
        """First-order wavefunctions sorted by energy."""
        return self.psi_1st[:, self.sort_idx_1st]

    @cached_property
    def sort_idx_2nd(self) -> np.ndarray:
        """Sorting indices for second-order energies."""
        return np.argsort(self.energies_2nd)

    @cached_property
    def energies_2nd_sorted(self) -> np.ndarray:
        """Second-order energies sorted by energy."""
        return self.energies_2nd[self.sort_idx_2nd]

    @cached_property
    def psi_2nd_sorted(self) -> np.ndarray:
        """Second-order wavefunctions sorted by energy."""
        return self.psi_2nd[:, self.sort_idx_2nd]

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
        return psi.conj().T @ op @ psi

    def reorder_to_energy_basis(self, operator=None, order="2nd"):
        """
        Sorts either the first- or second-order energies & wavefunctions
        by ascending energy, and caches the sorted versions plus the sort index.

        If operator is provided (dim x dim), it is also reordered (rows & columns)
        in the same way (energy basis). The returned 'op_sorted' corresponds
        to re-labeling basis states in ascending-energy order.

        Args:
            order    : "1st" or "2nd" (which cached results to reorder)
            operator : optional (dim, dim) matrix in product basis.
                       If given, we reorder it to match the energy basis ordering.

        Returns:
            op_sorted  # if operator is provided
            or
            (E_sorted, psi_sorted)  # if no operator is provided
        """
        if order == "1st":
            E_sorted = self.energies_1st_sorted
            psi_sorted = self.psi_1st_sorted
            sort_idx = self.sort_idx_1st
        else:  # order == "2nd"
            E_sorted = self.energies_2nd_sorted
            psi_sorted = self.psi_2nd_sorted
            sort_idx = self.sort_idx_2nd

        if operator is not None:
            op_sorted = operator[sort_idx, :][:, sort_idx]
            return E_sorted, psi_sorted, op_sorted
        else:
            return E_sorted, psi_sorted

    def reorder_to_product_basis(self, operator=None, order="2nd"):
        """
        Undo the ascending-energy ordering. We use the cached _sort_idx to invert the reordering.

        If operator is provided, we also reorder its rows & columns from the
        energy basis back to the original product basis labeling.

        Returns:
            op_unsorted  # if operator is provided
            or
            (E_unsorted, psi_unsorted)
        """
        if order == "1st":
            sort_idx = self.sort_idx_1st
            E_sorted = self.energies_1st_sorted
            psi_sorted = self.psi_1st_sorted
        else:
            sort_idx = self.sort_idx_2nd
            E_sorted = self.energies_2nd_sorted
            psi_sorted = self.psi_2nd_sorted

        # Invert the permutation
        inv_idx = np.empty_like(sort_idx)
        inv_idx[sort_idx] = np.arange(self.dim)

        if operator is not None:
            return operator[inv_idx, :][:, inv_idx]
        else:
            return E_sorted[inv_idx], psi_sorted[:, inv_idx]

class TwoBodyPerturbationDresser(PerturbationDresser):
    def __init__(self,         
                 energies_1: np.ndarray,  # shape (m,)
                energies_2: np.ndarray,  # shape (n,)
                n_op_1: np.ndarray,      # shape (m,m)
                n_op_2: np.ndarray,      # shape (n,n)
                g: float,
                ugly_fix_coefficient = 1
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
        super().__init__(dim, E0, V, ugly_fix_coefficient)

    def two_body_first_order_perturbation_vectorized(self) -> tuple[np.ndarray, np.ndarray]:
        return self.energies_1st, self.psi_1st
    
    def two_body_second_order_perturbation_vectorized(self) -> tuple[np.ndarray, np.ndarray]:
        return self.energies_2nd, self.psi_2nd


