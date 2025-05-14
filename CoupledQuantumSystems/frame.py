# CoupledQuantumSystems/frame.py
"""
Minimal rotating–frame helper (inspired by Qiskit Dynamics).
– Works with diagonal or full Hermitian frame operators.
– Provides fast   to_frame(op)   and   from_frame(op)   utilities.
"""
from dataclasses import dataclass
import numpy as np
import qutip

@dataclass
class RotatingFrame:
    H_frame: qutip.Qobj            # Hermitian
    evals:   np.ndarray            # eigenvalues (rad/s)
    U:       qutip.Qobj            # unitary eigenbasis    |ϕ_j⟩  (columns)

    @classmethod
    def from_operator(cls, Hf: qutip.Qobj, tol=1e-12):
        """Diagonalise H_frame if necessary and cache   evals, U."""
        if Hf.isdiag:
            evals = np.real(np.diag(Hf.full()))
            U     = qutip.qeye(Hf.dims[0])
        else:
            evals, U = Hf.eigenstates()
            evals = np.real(evals)
            U     = qutip.Qobj(qutip.operator_to_matrix(U))
        # Hermiticity check
        assert np.allclose(Hf.full().H, Hf.full(), atol=tol), "H_frame not Hermitian"
        return cls(H_frame=Hf, evals=evals, U=U)

    # ---------- basis transforms ----------
    def to_frame_basis(self, A: qutip.Qobj) -> qutip.Qobj:
        return (self.U.dag() * A * self.U).tidyup(1e-14)

    def from_frame_basis(self, Af: qutip.Qobj) -> qutip.Qobj:
        return (self.U * Af * self.U.dag()).tidyup(1e-14)

    # ---------- Bohr‑frequency matrix  ν_{jk} (Hz) ----------
    @property
    def bohr_freqs(self):
        return (self.evals[:, None] - self.evals[None, :]) / (2*np.pi)  # Hz

def static_rwa(frame, drift_op: qutip.Qobj, cutoff_freq: float):
    """
    Return a drift operator with all matrix elements whose frame–frequency
    |(d_j-d_k)/2π| > cutoff_freq [Hz] set to zero.
    """
    ν_jk = frame.bohr_freqs                             # Hz
    Hf   = frame.to_frame_basis(drift_op)               # frame basis
    Hf  -= qutip.Qobj(np.diag(frame.evals))             # subtract frame diag
    Hf   = qutip.Qobj(Hf.full() * (np.abs(ν_jk) < cutoff_freq)).tidyup(1e-14)
    return frame.from_frame_basis(Hf)                   # back to lab basis