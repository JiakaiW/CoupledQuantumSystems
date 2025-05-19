import numpy as np
import qutip
import sympy as sp 
from .drive_symbo import DriveTermSymbolic, JAX_AVAILABLE
from .pulse_shapes_symbo import gaussian_pulse, square_pulse_with_rise_fall
from . import pulse_shapes_symbo as pss
import matplotlib.pyplot as plt

# Imports from user example
import pickle
from tqdm import tqdm
from copy import deepcopy
import scqubits
from .systems import TransmonOscillatorSystem
from .evo import ODEsolve_and_post_process

def test_gaussian_pulse_and_simulation():
    # # Gaussian pulse part (mostly for reference, simulation uses square pulse)
    # sigma_x_op = qutip.sigmax()
    # expr_gauss, defaults_gauss, params_doc_gauss = gaussian_pulse()
    # drive_gauss = DriveTermSymbolic(
    #     driven_op=sigma_x_op,
    #     modulation_freq=5.0, 
    #     symbolic_expr=expr_gauss,
    #     symbolic_params={
    #         pss.amp: 0.1,
    #         pss.t_duration: 100,
    #         pss.t_start: 0,
    #         pss.how_many_sigma: 6,
    #         pss.normalize: True
    #     },
    #     pulse_id="gaussian_test_ref"
    # )
    # tlist_gauss = np.linspace(0, 200, 200) 
    # envelope_gauss = drive_gauss.pulse_shape_func(tlist_gauss, None, math=np)
    # coeff_gauss = drive_gauss.numpy_coeff()
    # full_signal_gauss = coeff_gauss(tlist_gauss)
    # print("Gaussian pulse envelope and full signal calculated (for reference).")

    # --- Simulation setup from user example --- 
    max_ql = 2
    max_ol = 3
    t_tot = 3.0 # Made float for linspace
    amp_val = 0.1 # Renamed from amp to amp_val to avoid conflict with pss.amp
    dt_val = 0.222 # Define dt for the simulation and pulse
    num_points = 4
    tlist_sim = np.linspace(0, t_tot, num_points)

    ncut = 110
    tmon_EJ = 22.867925499169555
    tmon_EC = 0.2492658230114663
    tmon_ng = 0.3
    Er = 20.88428612709732
    g = 0.599684130648041
    kappa_val = 0.022

    qbt = scqubits.Transmon(EJ=tmon_EJ, EC=tmon_EC, ng=tmon_ng, ncut=ncut, truncated_dim=max_ql)
    osc = scqubits.Oscillator(E_osc=Er, truncated_dim=max_ol, l_osc=1.0)

    system = TransmonOscillatorSystem(
        qbt=qbt,
        osc=osc,
        g_strength=g,
        kappa=kappa_val,
        products_to_keep=[[ql, ol] for ql in range(max_ql) for ol in range(max_ol)],
    )
    
    if not system.product_to_dressed:
        print("Warning: system.product_to_dressed is empty. Generating default mapping.")
        system.generate_default_mapping()
    
    key_00 = (0,0)
    key_01 = (0,1)
    key_10 = (1,0)
    key_11 = (1,1)

    if not all(k in system.product_to_dressed for k in [key_00, key_01, key_10, key_11]):
        raise KeyError(f"One or more keys ({key_00}, {key_01}, {key_10}, {key_11}) not in system.product_to_dressed: {system.product_to_dressed}")

    w_d = ((system.evals[system.product_to_dressed[key_01]] - system.evals[system.product_to_dressed[key_00]]) + \
           (system.evals[system.product_to_dressed[key_11]] - system.evals[system.product_to_dressed[key_10]])) / 2

    e_ops = [qutip.ket2dm(qutip.basis(max_ql * max_ol, system.product_to_dressed[(ql, ol)])) 
             for ql in range(max_ql) for ol in range(max_ol)] + [system.a.dag() * system.a]

    square_envelope_expr, defaults_envelope_params, params_doc_sq = square_pulse_with_rise_fall()
    
    # Ensure all default symbols are sympy.Symbol objects
    # For square_pulse_with_rise_fall, the primary parameters defining its shape are t_start, t_rise, t_square, amp.
    # It does not inherently use a 't_duration' symbol in its expression.
    # However, ScalableSymbolicPulse requires a 'duration' argument.
    # This duration should be the total extent of this specific pulse envelope.
    
    current_t_rise = 1e-13
    current_t_square = t_tot # Make the square part last for the whole simulation for this test
    # current_t_square = 1e9 # User's original very long pulse

    drive_params_sym = {
        pss.t_start: 0.0,
        pss.t_rise: current_t_rise,
        pss.t_square: current_t_square, 
        pss.amp: amp_val,
    }

    # Calculate the effective total duration for THIS pulse instance
    # This is what ScalableSymbolicPulse needs as its 'duration' parameter.
    # We will map our conceptual total duration (derived from components) to pss.t_duration 
    # so that to_qiskit_signal_jax can find it and pass it to ScalableSymbolicPulse.
    effective_pulse_duration = 2 * current_t_rise + current_t_square
    drive_params_sym[pss.t_duration] = effective_pulse_duration

    drive_term_sim = DriveTermSymbolic(
        driven_op=system.driven_operator,
        symbolic_expr=square_envelope_expr,
        symbolic_params=drive_params_sym, # Use the corrected params
        modulation_freq=w_d,
        pulse_id="square_pulse_sim_test", # Added pulse_id for clarity
        dt=dt_val # Pass dt to DriveTermSymbolic
    )

    initial_state_qobj = qutip.ket2dm(qutip.basis(max_ql * max_ol, system.product_to_dressed[key_00]))

    print(f"Running ODEsolve_and_post_process with method qiskit_dynamics...")
    print(f"  Initial state type: {type(initial_state_qobj)}")
    print(f"  Static Hamiltonian type: {type(system.diag_hamiltonian)}")
    print(f"  Drive terms: {[drive_term_sim]}")
    print(f"  C_ops: {[kappa_val * system.a]}")
    print(f"  E_ops count: {len(e_ops)}")
    print(f"  Rotating frame type: {type(system.diag_hamiltonian)}")
    print(f"  RWA carrier freqs: {[w_d]}")
    print(f"  tlist length: {len(tlist_sim)}")

    result = ODEsolve_and_post_process(
        y0=initial_state_qobj,
        tlist=tlist_sim,
        static_hamiltonian=system.diag_hamiltonian,
        drive_terms=[drive_term_sim],
        c_ops=[kappa_val * system.a],
        e_ops=e_ops,
        store_states=True,
        method='qiskit_dynamics',
        qiskit_solver_method='jax_odeint',
        rotating_frame=system.diag_hamiltonian,
        rwa_carrier_freqs=[w_d],
        print_progress=True
    )

    print("Simulation finished.")
    if result:
        print(f"Result contains {len(result.states)} states.")
        if result.expect:
            print(f"Result expectation values shape: {np.array(result.expect).shape}")

if __name__ == "__main__":
    test_gaussian_pulse_and_simulation() 