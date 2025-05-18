"""Fluxonium qubit implementation and analysis tools.

This module provides a specialized implementation of a fluxonium qubit system,
including methods for calculating transition rates and generating drive terms.
"""

import numpy as np
import qutip
import scqubits

from CoupledQuantumSystems.drive import *
from CoupledQuantumSystems.noise import *
from CoupledQuantumSystems.qobj_manip import *
from CoupledQuantumSystems.systems import QuantumSystem

class gfIFQ(QuantumSystem):
    """Fluxonium qubit implementation with additional analysis capabilities.

    This class extends the basic Fluxonium implementation to include methods for
    calculating transition rates and generating drive terms for quantum operations.

    Attributes:
        EJ (float): Josephson energy.
        EC (float): Charging energy.
        EL (float): Inductive energy.
        cutoff (int): Number of energy levels to consider.
        flux (float): External magnetic flux.
    """
    def __init__(self,
                 EJ,
                 EC,
                 EL,
                 flux=0, truncated_dim=20) -> None:
        self.fluxonium = scqubits.Fluxonium(EJ=EJ,
                                            EC=EC,
                                            EL=EL,
                                            flux=flux, cutoff=110,
                                            truncated_dim=truncated_dim)
        self.truncated_dim = truncated_dim
        self.evals = self.fluxonium.eigenvals(evals_count=truncated_dim)
        self.diag_hamiltonian = qutip.Qobj(2 * np.pi * np.diag(self.evals))
        self.phi_tabel = self.fluxonium.matrixelement_table(
            'phi_operator', evals_count=truncated_dim)
        self.n_tabel = self.fluxonium.matrixelement_table(
            'n_operator', evals_count=truncated_dim)

    def get_T1_c_ops(self,
                  temp_in_mK,
                  Q_cap,
                  one_over_f_flux_noise_amplitude) -> None:
        # array element [i,j] means transition rate from j to i
        dielectric_T1_array = np.full(
            (self.truncated_dim, self.truncated_dim), np.inf)
        one_over_f_T1_array = np.full(
            (self.truncated_dim, self.truncated_dim), np.inf)
        EL = self.fluxonium.EL
        EC = self.fluxonium.EC
        # T1
        for i in range(self.truncated_dim):
            for j in range(self.truncated_dim):
                if i == j:
                    continue
                freq = (self.evals[i]-self.evals[j]) * 2 * np.pi
                phi_ele = self.phi_tabel[i, j]
                dielectric_T1_array[j, i] = 1 / (np.abs(phi_ele)**2 * diel_spectral_density(
                    freq, EC, temp_in_mK, Q_cap=Q_cap))
                one_over_f_T1_array[j, i] = 1 / (np.abs(phi_ele)**2 * one_over_f_spectral_density(
                    freq, EL, one_over_f_flux_noise_amplitude))
        with np.errstate(divide='ignore', invalid='ignore'):
            T1_array = 1/(1/dielectric_T1_array + 1/one_over_f_T1_array)
        c_ops = qutip.Qobj(1/np.sqrt(T1_array))
        return c_ops

    def get_Tphi_c_ops(self,
                  one_over_f_flux_noise_amplitude) -> None:
        Tphi_array = np.zeros(shape=(self.truncated_dim,))
        # Tphi
        Tphi_array[0] = np.inf
        for ql in range(1,self.truncated_dim):
            Tphi_array[ql] = T_phi(
                second_order_derivative=second_order_derivative(partial(
                    get_fluxonium_frequency_with_2pi,
                    EJ=self.fluxonium.EJ,
                    EC=self.fluxonium.EC,
                    EL=self.fluxonium.EL,
                    i=0,j=ql
                    ),x0=0),
                one_over_f_flux_noise_amplitude=one_over_f_flux_noise_amplitude
            )
        c_ops = qutip.Qobj(np.diag(1/np.sqrt((Tphi_array))))
        return c_ops

    def get_STIRAP_drive_terms(self,
                               i,
                               j,
                               k,
                               t_stop,
                               Rabi_freqij=1e-1,
                               Rabi_freqjk = 1e-1,
                               detuning_ij = 0,
                               detuning_jk = 0,
                               t_start=0,
                               phi=0
                               ):
        amp_ij = Rabi_freqij / np.abs(self.n_tabel[i, j])
        amp_jk = Rabi_freqjk / np.abs(self.n_tabel[j, k])
        drive_terms = [
            DriveTerm(
                driven_op=qutip.Qobj(
                    self.fluxonium.n_operator(energy_esys=True)),
                pulse_shape_func=STIRAP_envelope,
                pulse_id='stoke',  # Stoke is the first pulse, pump is the second
                modulation_freq=np.abs(self.evals[k]-self.evals[j]) - detuning_ij,
                phi=phi,
                pulse_shape_args={
                    'amp': amp_jk,  # Without 2pi
                    't_stop': t_stop,
                    'stoke': True,
                    't_start': t_start,
                },
            ),
            DriveTerm(
                driven_op=qutip.Qobj(
                    self.fluxonium.n_operator(energy_esys=True)),
                pulse_shape_func=STIRAP_envelope,
                pulse_id='pump',
                modulation_freq=np.abs(self.evals[j]-self.evals[i]) - detuning_jk,
                phi=phi,
                pulse_shape_args={
                    'amp': amp_ij,  # Without 2pi
                    't_stop': t_stop,
                    'stoke': False,
                    't_start': t_start,
                },
            ),
        ]
        return drive_terms

    def get_Raman_drive_terms(self,
                              i,
                              j,
                              k,
                              t_duration,
                              shape:str,
                              detuning = None,
                              detuning1 = None,
                              detuning2 = None,
                              amp_scaling_factor= 1,
                              amp1_scaling_factor = 1,
                              amp2_scaling_factor = 1,
                              t_start=0,
                              phi=0
                              ):
        if detuning1 is None or detuning2 is None:
            if detuning is None:
                raise Exception('no detuning provided') 
            detuning1 = detuning
            detuning2 = detuning
        if shape == 'sin^2':
            # area =  amp= 2*np.pi /t_duration
            amp_ij = amp_scaling_factor*amp1_scaling_factor * np.pi / \
                t_duration / np.abs(self.n_tabel[i, j])
            amp_jk = amp_scaling_factor*amp2_scaling_factor*np.pi / \
                t_duration / np.abs(self.n_tabel[j, k])
            drive_terms = [
                DriveTerm(
                    driven_op=qutip.Qobj(
                        self.fluxonium.n_operator(energy_esys=True)),
                    pulse_shape_func=sin_squared_pulse_envelope,
                    pulse_id='ij',
                    modulation_freq=np.abs(self.evals[k]-self.evals[j])-detuning1,
                    phi=phi,
                    pulse_shape_args={
                        'amp': amp_jk,  # Without 2pi
                        't_duration': t_duration,
                        't_start': t_start,
                        'phi': phi
                    },
                ),
                DriveTerm(
                    driven_op=qutip.Qobj(
                        self.fluxonium.n_operator(energy_esys=True)),
                    pulse_shape_func=sin_squared_pulse_envelope,
                    pulse_id='jk',
                    modulation_freq=np.abs(self.evals[j]-self.evals[i])-detuning2,
                    phi=phi,
                    pulse_shape_args={
                        'amp': amp_ij,  # Without 2pi
                        't_duration': t_duration,
                        't_start': t_start,
                        'phi': phi
                    },
                ),
            ]
            return drive_terms
        
        elif shape == 'gaussian':
            how_many_sigma = 6
            sigma = t_duration/6
            # area =  amp= 2*np.pi /t_duration
            amp_ij = amp_scaling_factor*amp1_scaling_factor * np.sqrt(np.pi)/np.sqrt(2) / sigma / np.abs(self.n_tabel[i, j])
            amp_jk = amp_scaling_factor*amp2_scaling_factor* np.sqrt(np.pi)/np.sqrt(2) / sigma / np.abs(self.n_tabel[j, k])
            drive_terms = [
                DriveTerm(
                    driven_op=qutip.Qobj(
                        self.fluxonium.n_operator(energy_esys=True)),
                    pulse_shape_func=gaussian_pulse_envelope,
                    pulse_id='ij',
                    modulation_freq=np.abs(self.evals[j]-self.evals[i])-detuning1,
                    phi=phi,
                    pulse_shape_args={
                        'amp': amp_ij,  # Without 2pi
                        't_duration': t_duration,
                        'how_many_sigma': how_many_sigma,
                        'normalize':True,
                    },
                ),
                DriveTerm(
                    driven_op=qutip.Qobj(
                        self.fluxonium.n_operator(energy_esys=True)),
                    pulse_shape_func=gaussian_pulse_envelope,
                    pulse_id='jk',
                    modulation_freq=np.abs(self.evals[k]-self.evals[j])-detuning2,
                    phi=phi,
                    pulse_shape_args={
                        'amp': amp_jk,  # Without 2pi
                        't_duration': t_duration,
                        'how_many_sigma': how_many_sigma,
                        'normalize':True,
                   },
                ),
            ]
            return drive_terms

    def get_Raman_DRAG_drive_terms(self,
                              i,
                              j,
                              k,
                              t_duration,
                              shape:str,
                              detuning = None,
                              detuning1 = None,
                              detuning2 = None,
                              amp_scaling_factor = 1,
                              amp1_scaling_factor = 1,
                              amp2_scaling_factor = 1,
                              amp1_correction_scaling_factor = 0.05,
                              amp2_correction_scaling_factor = 0.05,
                              t_start=0,
                              phi=0
                              ):
        if detuning1 is None or detuning2 is None:
            if detuning is None:
                raise Exception('no detuning provided') 
            detuning1 = detuning
            detuning2 = detuning
        if shape == 'sin^2':
            # area =  amp= 2*np.pi /t_duration
            amp_ij = amp_scaling_factor*amp1_scaling_factor * np.pi / \
                t_duration / np.abs(self.n_tabel[i, j])
            amp_jk = amp_scaling_factor*amp2_scaling_factor*np.pi / \
                t_duration / np.abs(self.n_tabel[j, k])
            drive_terms = [
                DriveTerm(
                    driven_op=qutip.Qobj(
                        self.fluxonium.n_operator(energy_esys=True)),
                    pulse_shape_func=sin_squared_DRAG_envelope,
                    pulse_id='ij',
                    modulation_freq=np.abs(self.evals[j]-self.evals[i])-detuning1,
                    phi=phi,
                    pulse_shape_args={
                        'amp': amp_ij,  # Without 2pi
                        'amp_correction': amp_ij*amp1_correction_scaling_factor,
                        't_duration': t_duration,
                        't_start': t_start,
                        'phi': phi
                    },
                ),
                DriveTerm(
                    driven_op=qutip.Qobj(
                        self.fluxonium.n_operator(energy_esys=True)),
                    pulse_shape_func=sin_squared_DRAG_envelope,
                    pulse_id='jk',
                    modulation_freq=np.abs(self.evals[k]-self.evals[j])-detuning2,
                    phi=phi,
                    pulse_shape_args={
                        'amp': amp_jk,  # Without 2pi
                        'amp_correction': amp_jk*amp2_correction_scaling_factor,
                        't_duration': t_duration,
                        't_start': t_start,
                        'phi': phi
                    },
                ),
            ]
            return drive_terms
        
        elif shape == 'gaussian':
            how_many_sigma = 6
            sigma = t_duration/6
            # area =  amp= 2*np.pi /t_duration
            amp_ij = amp_scaling_factor*amp1_scaling_factor * np.sqrt(np.pi)/np.sqrt(2) / sigma / np.abs(self.n_tabel[i, j])
            amp_jk = amp_scaling_factor*amp2_scaling_factor* np.sqrt(np.pi)/np.sqrt(2) / sigma / np.abs(self.n_tabel[j, k])
            drive_terms = [
                DriveTerm(
                    driven_op=qutip.Qobj(
                        self.fluxonium.n_operator(energy_esys=True)),
                    pulse_shape_func=gaussian_DRAG_pulse_envelope,
                    pulse_id='ij',
                    modulation_freq=np.abs(self.evals[j]-self.evals[i])-detuning1,
                    phi=phi,
                    pulse_shape_args={
                        'amp': amp_ij,  # Without 2pi
                        't_duration': t_duration,
                        'how_many_sigma': how_many_sigma,
                        'normalize':True,
                        'amp_correction_scaling_factor':amp1_correction_scaling_factor
                    },
                ),
                DriveTerm(
                    driven_op=qutip.Qobj(
                        self.fluxonium.n_operator(energy_esys=True)),
                    pulse_shape_func=gaussian_DRAG_pulse_envelope,
                    pulse_id='jk',
                    modulation_freq=np.abs(self.evals[k]-self.evals[j])-detuning2,
                    phi=phi,
                    pulse_shape_args={
                        'amp': amp_jk,  # Without 2pi
                        't_duration': t_duration,
                        'how_many_sigma': how_many_sigma,
                        'normalize':True,
                        'amp_correction_scaling_factor':amp2_correction_scaling_factor
                   },
                ),
            ]
            return drive_terms
        
    # def get_composite_STIRAP_drive_terms(self):
    #     # PHYSICAL REVIEW A 87, 043418 (2013)
    #     pass

    # def get_STIRAP_with_DRAG_drive_terms(self):
    #     pass

    # def get_CD_STIRSAP_drive_terms(self):
    #     # CD is a form of Shortcuts-To-Adiabaticity
    #     # Antti Vepsäläinen et al. ,Superadiabatic population transfer in a three-level superconducting circuit.Sci. Adv.5,eaau5999(2019).DOI:10.1126/sciadv.aau5999

    #     # Introducing another 2-photon Counter-Adiabatic term
    #     pass

    # def get_STIRSAP_drive_terms(self):
    #     # Optimal control of stimulated Raman adiabatic passage in a superconducting qudit. npj Quantum Information volume 8, Article number: 9 (2022)
    #     # Optimize the detunings
    #     pass

    # def get_inertial_STIRAP_drive_terms(self):
        # Inertial geometric quantum logic gates D. Turyansky, O. Ovdat, R. Dann, Z. Aqua, R. Kosloff, B. Dayan, and A. Pick. Phys. Rev. Applied 21, 054033 – Published 17 May 2024
        pass

    def get_pi_pulse_drive_terms(self,
                                 i,
                                 j,
                                 t_square,
                                 amp=1e-2,
                                 ):

        drive_terms = [
            DriveTerm(
                driven_op=qutip.Qobj(
                    self.fluxonium.n_operator(energy_esys=True)),
                pulse_shape_func=square_pulse_with_rise_fall_envelope,
                pulse_id='pi',
                modulation_freq=self.evals[j]-self.evals[i],
                phi=0.0,
                pulse_shape_args={
                    'amp': amp,  # Without 2pi
                    't_square': t_square,
                },
            )
        ]
        return drive_terms