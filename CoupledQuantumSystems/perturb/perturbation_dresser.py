from __future__ import annotations
import numpy as np
from functools import cached_property
from .energy_index import EnergyIndex

class PerturbationDresser():
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

    @staticmethod
    def operator_in_perturbed_basis(
            psi: np.ndarray,  # (dim, dim), columns = |psi_n^(perturbed)>
            op: np.ndarray            # (dim, dim), operator in the original product basis
        ) -> np.ndarray:
        # Check shapes match
        dim = psi.shape[0]
        assert op.shape == (dim, dim), "Operator must match dimension of psi"
        
        # O_dressed = psi^\dagger @ op @ psi
        return psi.conj().T @ op @ psi

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

    @cached_property
    def energy_idx_1st(self) -> EnergyIndex:
        return EnergyIndex(self.energies_1st_sorted, self.psi_1st_sorted, self.sort_idx_1st, self.dim)
    
    @cached_property
    def energy_idx_2nd(self) -> EnergyIndex:
        return EnergyIndex(self.energies_2nd_sorted, self.psi_2nd_sorted, self.sort_idx_2nd, self.dim)

    def reorder_to_energy_basis(self, order="2nd"):
        if order == "1st":
            return self.energy_idx_1st.reorder_to_energy_basis()
        else:  # order == "2nd"
            return self.energy_idx_2nd.reorder_to_energy_basis()

    def reorder_op_to_energy_basis(self, operator: np.ndarray, order="2nd"):
        if order == "1st":
            return self.energy_idx_1st.reorder_op_to_energy_basis(operator)
        else:  # order == "2nd"
            return self.energy_idx_2nd.reorder_op_to_energy_basis(operator)

    def reorder_to_product_basis(self, order="2nd"):
        if order == "1st":
            return self.energy_idx_1st.reorder_to_product_basis()
        else:
            return self.energy_idx_2nd.reorder_to_product_basis()

    def reorder_op_to_product_basis(self, operator: np.ndarray, order="2nd"):
        if order == "1st":
            return self.energy_idx_1st.reorder_op_to_product_basis(operator)
        else:
            return self.energy_idx_2nd.reorder_op_to_product_basis(operator)

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

    def two_body_first_order_perturbation(self) -> tuple[np.ndarray, np.ndarray]:
        return self.energies_1st, self.psi_1st
    
    def two_body_second_order_perturbation(self) -> tuple[np.ndarray, np.ndarray]:
        return self.energies_2nd, self.psi_2nd


