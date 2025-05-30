from .dresser_base import AbstractDresser
from __future__ import annotations
import numpy as np
from functools import lru_cache
from typing import Sequence

# nested commutator ---------------------------------------------------
def nested_comm(ops: Sequence[np.ndarray]) -> np.ndarray:
    out = ops[-1]
    for op in reversed(ops[:-1]):
        out = op @ out - out @ op
    return out

# factorial, cached ---------------------------------------------------
@lru_cache(maxsize=None)
def fact(n: int) -> int:
    return 1 if n in (0, 1) else n * fact(n - 1)

# BCH expansion -------------------------------------------------------
def bch_expand(S: np.ndarray, O: np.ndarray, order: int) -> np.ndarray:
    O_dressed = O.copy()
    ad = O.copy()
    for k in range(1, order + 1):
        ad = S @ ad - ad @ S
        O_dressed += ad / fact(k)
    return O_dressed

# build Schrieffer–Wolff generator -----------------------------------
def sw_generator(
    Omega: dict[tuple[int, int], complex],
    Delta: dict[tuple[int, int], float],
    dim: int,
) -> np.ndarray:
    S = np.zeros((dim, dim), dtype=complex)
    for (m, n), Om in Omega.items():
        factor = Om / (2 * Delta[(m, n)])
        S[m, n] += factor
        S[n, m] -= np.conj(factor)
    return S

class BCHDresser(AbstractDresser):
    def __init__(self, bare_ops: dict[str, np.ndarray], order: int = 2):
        dim = next(iter(bare_ops.values())).shape[0]
        super().__init__(dim=dim)
        if order < 1:
            raise ValueError("order must be ≥ 1")
        self.order     = order
        self.bare_ops  = bare_ops          # e.g. {"n_t": n_t, "n_f": n_f}

    # ----------------------------------------------------------
    def _generator(self, Omega, Delta) -> np.ndarray:
        return sw_generator(Omega, Delta, dim=self.dim)

    def dress_operator(
        self,
        M_bare: np.ndarray,
        *,
        Omega: dict[tuple[int, int], complex],
        Delta: dict[tuple[int, int], float],
        projector: np.ndarray | None = None,
        **_,
    ) -> np.ndarray:
        S = self._generator(Omega, Delta)
        M_dressed = bch_expand(S, M_bare, self.order)
        return M_dressed if projector is None else projector @ M_dressed @ projector

    # energies / states (not typically needed for STIRAP loop) -
    def dress_energies(self, **_):
        raise NotImplementedError("Use perturbation engine for energies.")

    def dress_states(self, **_):
        raise NotImplementedError("Use perturbation engine for states.")

