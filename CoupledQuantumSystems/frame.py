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
    frame_diag:   np.ndarray            # eigenvalues (rad/s)
    frame_basis:       qutip.Qobj            # unitary eigenbasis    |ϕ_j⟩  (columns)
    cutoff_freq: float = 40      # GHz
    frame_freqs: np.ndarray = None
    dim: int = None
    frame_shift: qutip.Qobj = None
    @classmethod
    def from_operator(cls, Hf: qutip.Qobj, cutoff_freq: float = 40, tol=1e-12):
        """Diagonalise H_frame if necessary and cache   evals, U."""
        Hf_array = Hf.full()
        dim = Hf_array.shape[0]

        frame_diag, frame_basis = np.linalg.eigh(1j * (-1j)*Hf_array)   # anti‑Hermitian
        frame_diag = -1j * frame_diag
        # frame_basis = qutip.Qobj(frame_basis)
        Ucols = np.column_stack([ket.full() for ket in frame_basis])
        frame_basis = qutip.Qobj(Ucols)
        frame_freqs = (np.broadcast_to(frame_diag, (dim, dim)) - np.broadcast_to(frame_diag, (dim, dim)).T).imag 
        frame_shift = qutip.Qobj(1j*np.diag(frame_diag)) # This is real number
        return cls(frame_diag=frame_diag, frame_basis=frame_basis, frame_freqs=frame_freqs, cutoff_freq=cutoff_freq, dim=dim, frame_shift=frame_shift)

    # ---------- basis transforms ----------
    def into_frame_basis(self, op: qutip.Qobj) -> qutip.Qobj:
        return self.frame_basis.dag() * op * self.frame_basis

    def out_of_frame_basis(self, op: qutip.Qobj) -> qutip.Qobj:
        return self.frame_basis * op * self.frame_basis.dag()

    # TODO: add logic about transforming dissipators

    def conjugate_and_add(
            self,
            t,
            generator,
            op_to_add_in_fb= None,
            generator_in_frame_basis=False,
            return_in_frame_basis=False
            ):
        out = generator
        # 1. put the generator into frame basis (operator_into_frame_basis)
        if not generator_in_frame_basis:
            out = self.into_frame_basis(out)
        # get frame transformation matrix in diagonal basis
        # assumption that F is anti-Hermitian implies conjugation of
        # diagonal gives inversion
        exp_freq = np.exp(self.frame_diag * t)
        frame_mat = exp_freq.conj().reshape(self.dim, 1) * exp_freq

        out = out.full() * frame_mat

        if op_to_add_in_fb is not None:
            out = out + op_to_add_in_fb

        # if output is requested to not be in the frame basis, convert it
        if not return_in_frame_basis:
            out = self.out_of_frame_basis(out)
        return out

    def generator_into_frame(self,
                            t,
                            generator,
                            generator_in_frame_basis=False,
                            return_in_frame_basis=False):
        return self.conjugate_and_add(
                                t=t,
                                generator=generator,
                                op_to_add_in_fb= -np.diag(self.frame_diag),
                                generator_in_frame_basis=generator_in_frame_basis,
                                return_in_frame_basis=return_in_frame_basis
                                )

    def static_rwa(self, static_op: qutip.Qobj):# suppose the input is hamiltonian (hermitian)
        static_generator_in_frame_basis = self.generator_into_frame(
                t=0,
                generator=-1j*static_op,
                return_in_frame_basis=True,
            )

        static_op = 1j*static_generator_in_frame_basis + self.frame_shift # This result should actually be called op instead of generator
        static_op_rwa = qutip.Qobj(static_op.full() *(abs(self.frame_freqs) < self.cutoff_freq).astype(int))
        return self.out_of_frame_basis(static_op_rwa)                   # back to lab basis
    
    def rwa_transform_drive_terms(
            self,
            drive_terms: List[DriveTerm],
    ) -> List[DriveTerm]:
        """
        Basically replicating qiskit dynamics get_rwa_operators()
        """
        out_terms: List[DriveTerm] = []

        def _is_zero(M: qutip.Qobj): return M.data.nnz == 0

        for term in drive_terms: # each term has one operator, for which we output two terms
            generator = -1j*term.driven_op
            generator_in_frame_basis = self.into_frame_basis(generator)
            op  = generator_in_frame_basis.full()

            ω_d = term.pulse_shape_args['w_d']          # GHz
            φ   = term.pulse_shape_args.get('phi', 0.0)
            keep_pos = np.abs(+ω_d + self.frame_freqs) < self.cutoff_freq
            keep_neg = np.abs(-ω_d + self.frame_freqs) < self.cutoff_freq
            if not (keep_pos.any() or keep_neg.any()):
                continue

            op_pos = qutip.Qobj(op * keep_pos)
            op_neg = qutip.Qobj(op * keep_neg)
            op_real   = 0.5*(op_pos + op_neg)
            op_imag   = 0.5j*(op_pos - op_neg)

            if not _is_zero(op_real):
                out_terms.append(
                    DriveTerm(
                        driven_op       = 1j*self.out_of_frame_basis(op_real),
                        pulse_shape_func= term.pulse_shape_func,
                        pulse_shape_args={**term.pulse_shape_args},
                        pulse_id        =(term.pulse_id or "")+"_rwa"
                    )
                )
            if not _is_zero(op_imag):
                out_terms.append(
                    DriveTerm(
                        driven_op       = 1j*self.out_of_frame_basis(op_imag),
                        pulse_shape_func= term.pulse_shape_func,
                        pulse_shape_args={**term.pulse_shape_args,'phi':φ-np.pi/2},
                        pulse_id        =(term.pulse_id or "")+"_rwa_q"
                    )
                )
        return out_terms
