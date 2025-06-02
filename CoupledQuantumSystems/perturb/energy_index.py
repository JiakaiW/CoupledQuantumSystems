"""
EnergyIndexMixin
================
Utility mix-in that handles the bookkeeping needed to hop
between *product-basis* ordering (whatever order your Hamiltonian
comes in) and *ascending-energy* ordering.

All classes that inherit must define:

    self.E0  -- shape (dim,) ndarray of unperturbed energies
    self.dim -- int, Hilbert-space dimension
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

@dataclass
class EnergyIndex:
    energies_sorted: np.ndarray
    psi_sorted: np.ndarray
    sort_idx: np.ndarray
    dim: int

    def __post_init__(self):
        self.inv_idx = np.empty_like(self.sort_idx)
        self.inv_idx[self.sort_idx] = np.arange(self.dim)

    def reorder_to_energy_basis(self):
        return self.energies_sorted, self.psi_sorted
    
    def reorder_op_to_energy_basis(self, operator: np.ndarray):
        return operator[self.sort_idx, :][:, self.sort_idx]
    
    def reorder_to_product_basis(self):
        return self.energies_sorted[self.sort_idx], self.psi_sorted[:, self.sort_idx]
    
    def reorder_op_to_product_basis(self, operator: np.ndarray):
        return operator[self.inv_idx, :][:, self.inv_idx]