import numpy as np

class Perturbation:
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
    """

    def __init__(self, dim: int, E0: np.ndarray, V: np.ndarray, ugly_fix_coefficient = 1):
        """
        Args:
            dim : dimension of the Hilbert space
            E0  : shape (dim,), unperturbed energies in the chosen basis
            V   : shape (dim, dim), the perturbation matrix in that same basis
        """
        self.dim = dim
        self.E0 = E0
        self.V = V

        # Internal caches for first- and second-order computations
        self._energies_1st = None       # shape (dim,)
        self._psi_1st = None            # shape (dim, dim)
        self._energies_2nd = None       # shape (dim,)
        self._psi_2nd = None            # shape (dim, dim)

        # We'll also keep "sorted" versions + the sort indices
        self._energies_1st_sorted = None
        self._psi_1st_sorted = None
        self._sort_idx_1st = None

        self._energies_2nd_sorted = None
        self._psi_2nd_sorted = None
        self._sort_idx_2nd = None

        self.ugly_fix_coefficient = ugly_fix_coefficient

    @staticmethod
    def embed_operator(op: np.ndarray, dims_list: list[int], where_to_embed: int) -> np.ndarray:
        op_list = [np.eye(dim, dtype=op.dtype) for dim in dims_list]
        op_list[where_to_embed] = op
        return np.kron(*op_list)


    def first_order_perturbation(self):
        # check cache
        if self._energies_1st is not None and self._psi_1st is not None:
            return self._energies_1st, self._psi_1st

        # (A) First-order energies
        E1 = np.diag(self.V)
        self._energies_1st = self.E0 + E1  # shape (dim,)

        # (B) Build denominators
        E0_row = self.E0.reshape(1, self.dim)
        E0_col = self.E0.reshape(self.dim, 1)
        denom = E0_row - E0_col
        np.fill_diagonal(denom, np.inf)

        # (C) Coeffs
        coeffs = self.V / denom

        # (D) Wavefunctions up to 1st order (unnormalized)
        self._psi_1st = np.eye(self.dim, dtype=np.complex128) + coeffs / self.ugly_fix_coefficient

        # Normalize the wavefunctions
        norms = np.linalg.norm(self._psi_1st, axis=0)  # compute norms for all columns at once
        self._psi_1st /= norms[np.newaxis, :]  # broadcast division across all rows

        return self._energies_1st, self._psi_1st
    
    def second_order_perturbation(self):
        if self._energies_1st is None or self._psi_1st is None:
            self.first_order_perturbation()

        # check cache
        if self._energies_2nd is not None and self._psi_2nd is not None:
            return self._energies_2nd, self._psi_2nd

        # (1) Second-order energy shift
        abs_V_sq = np.abs(self.V)**2
        E0_row = self.E0.reshape(1, self.dim)
        E0_col = self.E0.reshape(self.dim, 1)
        denom = E0_row - E0_col
        np.fill_diagonal(denom, np.inf)

        second_order_matrix = abs_V_sq / denom
        E2 = np.sum(second_order_matrix, axis=0)
        self._energies_2nd = self._energies_1st + E2  # E(0+1+2)

        # (2) Second-order wavefunction corrections
        #   |psi_n^(2)> = ∑_{m ≠ n}  (⟨m|V|psi_n^(1)> / [E0[n]-E0[m]])  |m>
        #
        # We'll do this in a vectorized way by:
        #   numerator = V @ psi1st   (matrix-matrix product, shape (dim, dim))
        # so column n is V * (psi_n^(1))
        #   => numerator[m,n] = ∑_p V[m,p]*psi1st[p,n] = ⟨m|V|psi_n^(1)⟩
        #
        # Then denom[m,n] = E0[n] - E0[m], same shape (dim, dim).
        # => c2[m,n] = numerator[m,n] / denom[m,n] for m != n.

        numerator = self.V @ self._psi_1st  # shape (dim, dim)
        # denom = E0_row - E0_col
        # np.fill_diagonal(denom, np.inf)

        psi2 = numerator / denom  # shape (dim, dim)
        # For safety, zero out the diagonal so we never add infinite self-term
        np.fill_diagonal(psi2, 0.0)

        # (D) The total wavefunction up to second order
        psi_up_to_2 = self._psi_1st + psi2 / self.ugly_fix_coefficient

        # Normalize the wavefunctions
        norms = np.linalg.norm(psi_up_to_2, axis=0)  # compute norms for all columns at once
        psi_up_to_2 /= norms[np.newaxis, :]  # broadcast division across all rows

        self._psi_2nd = psi_up_to_2 

        return self._energies_2nd, self._psi_2nd

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

    def generate_1st_sorting(self):
        """Generate sorting indices and sorted arrays for first-order perturbation."""
        if self._energies_1st is None or self._psi_1st is None:
            self.first_order_perturbation()

        sort_idx = np.argsort(self._energies_1st)
        E_sorted = self._energies_1st[sort_idx]
        psi_sorted = self._psi_1st[:, sort_idx]

        # Cache the results
        self._sort_idx_1st = sort_idx
        self._energies_1st_sorted = E_sorted
        self._psi_1st_sorted = psi_sorted

        return sort_idx, E_sorted, psi_sorted

    def generate_2nd_sorting(self):
        """Generate sorting indices and sorted arrays for second-order perturbation."""
        if self._energies_2nd is None or self._psi_2nd is None:
            self.second_order_perturbation()

        sort_idx = np.argsort(self._energies_2nd)
        E_sorted = self._energies_2nd[sort_idx]
        psi_sorted = self._psi_2nd[:, sort_idx]

        # Cache the results
        self._sort_idx_2nd = sort_idx
        self._energies_2nd_sorted = E_sorted
        self._psi_2nd_sorted = psi_sorted

        return sort_idx, E_sorted, psi_sorted

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
            # Check if already sorted
            if self._sort_idx_1st is not None and self._energies_1st_sorted is not None:
                E_sorted = self._energies_1st_sorted
                psi_sorted = self._psi_1st_sorted
                sort_idx = self._sort_idx_1st
            else:
                sort_idx, E_sorted, psi_sorted = self.generate_1st_sorting()
        else:  # order == "2nd"
            # Check if already sorted
            if self._sort_idx_2nd is not None and self._energies_2nd_sorted is not None:
                E_sorted = self._energies_2nd_sorted
                psi_sorted = self._psi_2nd_sorted
                sort_idx = self._sort_idx_2nd
            else:
                sort_idx, E_sorted, psi_sorted = self.generate_2nd_sorting()

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
            if self._sort_idx_1st is not None and self._energies_1st_sorted is not None:
                sort_idx = self._sort_idx_1st
                E_sorted = self._energies_1st_sorted
                psi_sorted = self._psi_1st_sorted
            else:
                sort_idx, E_sorted, psi_sorted = self.generate_1st_sorting()
        else:
            if self._sort_idx_2nd is not None and self._energies_2nd_sorted is not None:
                sort_idx = self._sort_idx_2nd
                E_sorted = self._energies_2nd_sorted
                psi_sorted = self._psi_2nd_sorted
            else:
                sort_idx, E_sorted, psi_sorted = self.generate_2nd_sorting()

        # Invert the permutation
        inv_idx = np.empty_like(sort_idx)
        inv_idx[sort_idx] = np.arange(self.dim)

        if operator is not None:
            return operator[inv_idx, :][:, inv_idx]
        else:
            return E_sorted[inv_idx], psi_sorted[:, inv_idx]

class TwoBodyPerturbation(Perturbation):
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
        return self.first_order_perturbation()
    
    def two_body_second_order_perturbation_vectorized(self) -> tuple[np.ndarray, np.ndarray]:
        return self.second_order_perturbation()


