from __future__ import annotations
import numpy as np
from functools import lru_cache

# factorial, cached ---------------------------------------------------
@lru_cache(maxsize=None)
def fact(n: int) -> int:
    return 1 if n in (0, 1) else n * fact(n - 1)

# BCH expansion -------------------------------------------------------
def bch_expand(S: np.ndarray, O: np.ndarray, order: int) -> np.ndarray:
    # A way to approximate how one matrix (an operator) transforms under the "action" of another matrix, (the hard-to-compute e^{S} O e^{-S} that contains matrix exponential)
    # this BCH expansion gives you a systematic way to do that without full matrix exponentials using the **BCH series**: e^S O e^{-S} = O + [S, O] + \frac{1}{2!}[S, [S, O]] + \frac{1}{3!}[S, [S, [S, O]]] + \cdots
    O_dressed = O.copy()
    ad = O.copy()
    for k in range(1, order + 1):
        ad = S @ ad - ad @ S
        O_dressed += ad / fact(k)
    return O_dressed

# build Schrieffer–Wolff generator -----------------------------------
def sw_generator(evals: np.ndarray, operator: np.ndarray) -> np.ndarray:
    '''
    Constructs the Schrieffer-Wolff generator matrix `S` (skew-Hermitian).
    This matrix is used in a similarity transformation e^S H e^{-S} to decouple
    or perturbatively treat off-diagonal couplings in a Hamiltonian.
    The generator S is constructed to first order in the coupling operator.

    The elements of S are given by S_mn = V_mn / (2 * (E_m - E_n)) for m != n,
    and S_mm = 0. Here, V is the 'operator' and E are the 'evals'.

    Inputs:
    * `evals`: 1D NumPy array of unperturbed eigenvalues (E_m).
    * `operator`: 2D NumPy array representing the coupling operator (V_mn).
                 Typically, this is the off-diagonal part of the Hamiltonian
                 in the basis where H0 (diagonal with `evals`) is diagonal.
    
    Returns:
    * `S`: 2D NumPy array, the skew-Hermitian generator matrix.
    
    The function identifies significant couplings (abs(operator[m,n]) > 1e-10)
    in the upper triangle (m < n), computes the corresponding S_mn factor,
    and sets S_nm = -conj(S_mn). Diagonal elements of S are zero.
    '''
    dim = operator.shape[0]
    if not (evals.shape == (dim,) and operator.shape == (dim, dim)):
        raise ValueError(
            f"Shape mismatch: evals should be ({dim},), operator ({dim},{dim}). "
            f"Got evals: {evals.shape}, operator: {operator.shape}"
        )

    S = np.zeros_like(operator, dtype=complex)

    # Indices for upper triangle (m < n)
    m_indices, n_indices = np.triu_indices(dim, k=1)

    if m_indices.size == 0: # Handles dim < 2 (no off-diagonal elements)
        return S

    # Extract corresponding operator elements and energy differences
    op_mn = operator[m_indices, n_indices].astype(complex)
    # Ensure evals are float for delta_mn to avoid type issues if evals are int
    delta_mn = evals[m_indices].astype(float) - evals[n_indices].astype(float)

    # Mask for significant couplings
    significant_mask = np.abs(op_mn) > 1e-10

    # Filter elements based on the mask
    op_mn_sig = op_mn[significant_mask]
    delta_mn_sig = delta_mn[significant_mask]
    
    m_sig = m_indices[significant_mask]
    n_sig = n_indices[significant_mask]

    if m_sig.size == 0: # No significant couplings to process
        return S
        
    # Calculate factors.
    # numpy's division by zero (if delta_mn_sig contains zeros where op_mn_sig is non-zero)
    # will result in 'inf' and a RuntimeWarning. This behavior is consistent with
    # the underlying physics where S would diverge for resonant couplings.
    factors_sig = op_mn_sig / (2 * delta_mn_sig)
    
    # Assign to S matrix
    S[m_sig, n_sig] = factors_sig
    S[n_sig, m_sig] = -np.conj(factors_sig) # Ensures S is skew-Hermitian
    
    return S

class BCHDresser():
    def __init__(self, bare_ops: dict[str, np.ndarray], order: int = 2):
        dim = next(iter(bare_ops.values())).shape[0]
        super().__init__(dim=dim)
        if order < 1:
            raise ValueError("order must be ≥ 1")
        self.order     = order
        self.bare_ops  = bare_ops          # e.g. {"n_t": n_t, "n_f": n_f}

    # ----------------------------------------------------------
    def _generator(self, evals: np.ndarray, operator: np.ndarray) -> np.ndarray:
        # The sw_generator function itself will validate shapes.
        # self.dim (from BCHDresser.__init__) could be used for additional assertions here if desired:
        # e.g. assert evals.shape[0] == self.dim and operator.shape == (self.dim, self.dim)
        return sw_generator(evals, operator)

    def dress_operator(
        self,
        M_bare: np.ndarray,
        *,
        evals: np.ndarray,             # Parameter changed from Omega
        operator: np.ndarray,          # Parameter changed from Delta
        projector: np.ndarray | None = None,
        **_,
    ) -> np.ndarray:
        S = self._generator(evals, operator) # Call updated
        M_dressed = bch_expand(S, M_bare, self.order)
        return M_dressed if projector is None else projector @ M_dressed @ projector
