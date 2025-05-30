# -------------------------------------------------------------
# Common abstract interface for any weak-coupling dresser
# -------------------------------------------------------------
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

class AbstractDresser(ABC):
    """Abstract API every dressing backend must satisfy."""

    def __init__(self, dim: int):
        self.dim = dim

    # -------- mandatory hooks ---------------------------------
    @abstractmethod
    def dress_operator(
        self,
        M_bare: np.ndarray,
        **context,
    ) -> np.ndarray:
        """Return e^{S} M e^{-S} (optionally projected)."""
        ...

    @abstractmethod
    def dress_energies(self, **context) -> np.ndarray:
        """Return dressed eigen-energies to the implemented order."""
        ...

    @abstractmethod
    def dress_states(self, **context) -> np.ndarray:
        """Return dressed eigen-vectors (columns)."""
        ...
