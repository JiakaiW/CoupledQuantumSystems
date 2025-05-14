# CoupledQuantumSystems/frame.py
"""
Minimal rotating–frame helper (inspired by Qiskit Dynamics).
– Works with diagonal or full Hermitian frame operators.
– Provides fast   to_frame(op)   and   from_frame(op)   utilities.
"""
from dataclasses import dataclass
import numpy as np
import qutip
from typing import  Callable, Dict, List
from CoupledQuantumSystems.drive import DriveTerm
@dataclass
class RotatingFrame:
    H_frame: qutip.Qobj            # Hermitian
    evals:   np.ndarray            # eigenvalues (rad/s)
    U:       qutip.Qobj            # unitary eigenbasis    |ϕ_j⟩  (columns)
    cutoff_freq: float = 40      # GHz
    @classmethod
    def from_operator(cls, Hf: qutip.Qobj, cutoff_freq: float = 40, tol=1e-12):
        """Diagonalise H_frame if necessary and cache   evals, U."""
        mat = Hf.full()
        if np.all(mat == np.diag(np.diagonal(mat))):
            evals = np.real(np.diag(mat))
            U     = qutip.qeye(Hf.dims[0])
        else:
            evals, U = Hf.eigenstates()
            evals = np.real(evals)
            U     = qutip.Qobj(qutip.operator_to_matrix(U))
        # Hermiticity check
        assert np.allclose(mat.T.conj(), mat, atol=tol), "H_frame not Hermitian"
        return cls(H_frame=Hf, evals=evals, U=U, cutoff_freq=cutoff_freq)

    # ---------- basis transforms ----------
    def to_frame_basis(self, A: qutip.Qobj) -> qutip.Qobj:
        return (self.U.dag() * A * self.U).tidyup(1e-14)

    def from_frame_basis(self, Af: qutip.Qobj) -> qutip.Qobj:
        return (self.U * Af * self.U.dag()).tidyup(1e-14)

    # ---------- Bohr‑frequency matrix  ν_{jk} (Hz) ----------
    @property
    def bohr_freqs(self):
        return (self.evals[:, None] - self.evals[None, :]) / (2*np.pi)  # GHz

    def static_rwa(self, drift_op: qutip.Qobj):
        """
        Return a drift operator with all matrix elements whose frame–frequency
        |(d_j-d_k)/2π| > cutoff_freq [GHz] set to zero.
        """
        ν_jk = self.bohr_freqs                             # Hz
        Hf   = self.to_frame_basis(drift_op)               # frame basis
        Hf  -= qutip.Qobj(np.diag(self.evals))             # subtract frame diag
        Hf   = qutip.Qobj(Hf.full() * (np.abs(ν_jk) < self.cutoff_freq)).tidyup(1e-14)
        return self.from_frame_basis(Hf)                   # back to lab basis
    
    def rwa_transform_drive_terms(
            self,
            drive_terms: List[DriveTerm],
    ) -> List[DriveTerm]:
        """
        Filter each DriveTerm operator in *frame basis* and split into
        real / quadrature components (Hermitian).
        """
        ν_jk = self.bohr_freqs
        out_terms: List[DriveTerm] = []

        def _is_zero(M: qutip.Qobj): return M.data.nnz == 0

        for term in drive_terms:
            ω_d = term.pulse_shape_args['w_d']          # GHz
            φ   = term.pulse_shape_args.get('phi', 0.0)
            Gf  = self.to_frame_basis(term.driven_op).full()

            keep_pos = np.abs(+ω_d + ν_jk) < self.cutoff_freq
            keep_neg = np.abs(-ω_d + ν_jk) < self.cutoff_freq
            if not (keep_pos.any() or keep_neg.any()):
                continue

            G_pos = qutip.Qobj(Gf * keep_pos)
            G_neg = qutip.Qobj(Gf * keep_neg)
            G_c   = 0.5*(G_pos + G_neg)
            G_s   = 0.5j*(G_pos - G_neg)

            if not _is_zero(G_c):
                out_terms.append(
                    DriveTerm(
                        driven_op       = self.from_frame_basis(G_c).tidyup(1e-14),
                        pulse_shape_func= term.pulse_shape_func,
                        pulse_shape_args={**term.pulse_shape_args,'w_d':0.0},
                        pulse_id        =(term.pulse_id or "")+"_rwa"
                    )
                )
            if not _is_zero(G_s):
                out_terms.append(
                    DriveTerm(
                        driven_op       = self.from_frame_basis(G_s).tidyup(1e-14),
                        pulse_shape_func= term.pulse_shape_func,
                        pulse_shape_args={**term.pulse_shape_args,'phi':φ-np.pi/2,'w_d':0.0},
                        pulse_id        =(term.pulse_id or "")+"_rwa_q"
                    )
                )
        return out_terms
    
    def _phase_op(self, t: float, sign: int = +1) -> qutip.Qobj:
        """
        Return U_phase(t) = diag(exp(sign * i * evals * t))
        sign = +1  →  exp(+i E_j t)   (appears in U†)
        sign = -1  →  exp(-i E_j t)   (appears in U)
        """
        phases = np.exp(1j * sign * self.evals * t)
        return qutip.Qobj(np.diag(phases), dims=self.H_frame.dims)

    # ------------ ket or density‑matrix into the rotating frame ----
    def state_to_frame(self, state: qutip.Qobj, t: float) -> qutip.Qobj:
        """
        Convert |ψ_lab(t)⟩  or  ρ_lab(t)  into the rotating‑frame picture:
            ψ_frame = U†(t) ψ_lab,   ρ_frame = U†(t) ρ_lab U(t)
        Works for kets and density matrices.
        """
        U_phase = self.U * self._phase_op(t, +1) * self.U.dag()   # U†(t)
        if state.isket:
            return (U_phase * state).unit()
        elif state.isoper:
            return U_phase * state * U_phase.dag()
        else:
            raise TypeError("state must be ket or density matrix")

    # ------------ back to the laboratory frame --------------------
    def state_from_frame(self, state_f: qutip.Qobj, t: float) -> qutip.Qobj:
        """
        Inverse transformation:
            ψ_lab = U(t) ψ_frame,   ρ_lab = U(t) ρ_frame U†(t)
        """
        U   = self.U * self._phase_op(t, -1) * self.U.dag()       # U(t)
        if state_f.isket:
            return (U * state_f).unit()
        elif state_f.isoper:
            return U * state_f * U.dag()
        else:
            raise TypeError("state must be ket or density matrix")