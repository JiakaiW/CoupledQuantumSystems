# import os

# os.environ['JAX_JIT_PJIT_API_MERGE'] = '0'
# import jax.numpy as jnp

# import numpy as np
# import qiskit.pulse
# from qiskit_dynamics.array import Array
# Array.set_default_backend('jax')
# from qiskit_dynamics import Solver
# from qiskit_dynamics.models import HamiltonianModel, LindbladModel
# from qiskit_dynamics.solvers.solver_classes import validate_and_format_initial_state
# import qutip
# import sympy as sym
# from tqdm import tqdm
# import matplotlib.pyplot as plt



# def run_jax_solve(w_d,
#                 static_hamiltonian: qutip.Qobj,   
#                 initial_state, 
#                 tlist, 
#                 c_ops = None,
#                 mode = 'gpu',
                
#                 amp = 0.004,
#                 t_stop=None,
#                 t_rise=None,
#                 driven_operator = None, # like a_trunc+a_trunc.dag()
#                 chunk_size=2 
#                 ):

#     signal_sample_dt = 0.1
    
#     if mode == 'cpu':
#         evaluation_mode =  "sparse"
#     elif mode == 'gpu':
#         evaluation_mode = "dense_vectorized" if  c_ops is not None else "dense"
#     else:
#         raise Exception(f'mode not supported: mode: {mode}')
    
#     ham_solver = Solver(
#         hamiltonian_operators=[qobj_to_Array(driven_operator)],
#         static_hamiltonian=2 * np.pi * qobj_to_Array(static_hamiltonian),
#         static_dissipators= [qobj_to_Array(op) for op in c_ops] if  c_ops is not None else None,
#         hamiltonian_channels=['d0'],
#         channel_carrier_freqs={'d0': w_d * 2 * np.pi},
#         dt=signal_sample_dt,
#         evaluation_mode = evaluation_mode
#     )

#     initial_state = qobj_to_Array(initial_state)
#     if c_ops != None:
#         initial_state = initial_state @ initial_state.conj().T  
#         if evaluation_mode == "dense_vectorized":
#             initial_state = initial_state.flatten(order='F')  

#     signals = get_qiskit_square_pulse_with_sin_squared_edges(
#                                             amp_without_2pi = amp,
#                                             t_rise = t_rise if t_rise is not None else 0,
#                                             t_stop = t_stop if t_stop is not None else tlist[-1]
#                                             )
#     if mode == 'cpu':
#         result = ham_solver.solve(
#                 y0=initial_state, 
#                 t_span=[0, tlist[-1]],
#                 signals=signals, 
#                 method='jax_odeint', 
#                 t_eval=tlist, 
#             )
#         tlist = result.tlist
#         states = result.y

#     elif mode == 'gpu':
#         ########################################################################################
#         #  
#         # For qiskit-dynamics' GPU solvers,
#         #  truncating through time is still needed as of Feb 18 2024, if the t_span used to
#         #    call ham_solver.solve os too large you get VRAM issue (setting a small max_dt won't save you)
#         #
#         ########################################################################################
    
#         if chunk_size >= 1:
#             ###################################
#             # Here, the tlist of the result is still the original tlist
#             ###################################
#             # TODO: this has an issue of t_eval not included in t_span, debug later
#             tlist_chunks = [tlist[i:i + chunk_size] for i in range(0, len(tlist), chunk_size)]
#             current_state = initial_state
#             chunk_results = []

#             for chunk in tqdm(tlist_chunks,desc=f"solving through chunks"):
#                 print()
#                 result = ham_solver.solve(
#                     y0=current_state, 
#                     t_span=[chunk[0], chunk[-1]],
#                     signals=signals, 
#                     method='jax_expm_parallel', 
#                     t_eval=jnp.linspace(chunk[0], chunk[-1], len(chunk)), 
#                     max_dt=1 
#                 )

#                 if chunk_results == []:
#                     chunk_results.extend(result.y)
#                 else:
#                     chunk_results.extend(result.y[1:])

#                 current_state = result.y[-1]

#             states=chunk_results
#         else:
#             ###################################
#             # Here, the tlist of the result is a new one, maybe I will change this in the future.
#             ###################################
#             current_state = initial_state
#             chunk_results = []
#             t_results = []
#             total_time = tlist[-1] - tlist[0]
#             num_intervals = int(len(tlist) / chunk_size)
#             interval_length = total_time / num_intervals

#             for i in tqdm(range(num_intervals),desc=f"solving through chunks"):
#                 t_start = tlist[0] + i * interval_length
#                 t_end = t_start + interval_length
#                 t_eval_interval = jnp.array([t_end])
#                 result = ham_solver.solve(
#                     y0=current_state,
#                     t_span=[t_start, t_end],
#                     signals=signals,
#                     method='jax_expm_parallel',
#                     t_eval=t_eval_interval,
#                     max_dt=1
#                 )

#                 chunk_results.append(result.y[-1])
#                 t_results.append(t_eval_interval[-1])
#                 current_state = result.y[-1] 
            
#             tlist=t_results
#             states=chunk_results
        
#     ode_result = qutip.solver.Result()
#     ode_result.times= tlist
#     ode_result.states= [qutip.Qobj(np.array(state)) for state in states]
#     return ode_result



# def get_qiskit_square_pulse_with_sin_squared_edges(amp_without_2pi = 0.03, t_rise = 15,t_stop = 100):
#         ########################################################################################
#         #  
#         # qiskit-dynamics has a different way of specifing a pulse
#         #
#         ########################################################################################
#     t_square =  t_stop - 2 * t_rise
#     signal_sample_dt = 0.1 # Sample rate of the backend in ns.
#     base_drive_amplitude = amp_without_2pi * 2 * np.pi
#     _t = sym.symbols('t')
#     _amp = sym.symbols('amp')
#     _duration = sym.symbols('duration')
#     with qiskit.pulse.build(name="square") as square:
#         if t_rise > 0:
#             qiskit.pulse.play(
#                 qiskit.pulse.library.ScalableSymbolicPulse(
#                     pulse_type="SinSquaredRingUp",
#                     duration= int(t_rise/signal_sample_dt), 
#                     amp=base_drive_amplitude, 
#                     angle=0,
#                     envelope= _amp * sym.sin(sym.pi * _t  / (2 * t_rise / signal_sample_dt) )**2,
#                 ), 
#                 qiskit.pulse.DriveChannel(0)
#             )
#         qiskit.pulse.play(
#             qiskit.pulse.Constant(
#                 duration = int(t_square/signal_sample_dt), 
#                 amp = base_drive_amplitude
#             ), 
#             qiskit.pulse.DriveChannel(0)
#         )
#         if t_rise > 0:
#             qiskit.pulse.play(
#                 qiskit.pulse.library.ScalableSymbolicPulse(
#                     pulse_type="SinSquaredRingDown",
#                     duration=int(t_rise/signal_sample_dt), 
#                     amp=base_drive_amplitude,
#                     angle=0, 
#                     envelope=  _amp* sym.sin(sym.pi * (_duration - _t ) / (2 * t_rise / signal_sample_dt))**2,
#                 ), 
#                 qiskit.pulse.DriveChannel(0)
#             )
#     return square

# def draw_qiskit_pulse(p, carrier_freq, t_tot):
#     p.draw()
#     signal_sample_dt = 0.1
#     from qiskit_dynamics.pulse import InstructionToSignals
#     converter = InstructionToSignals(signal_sample_dt, carriers={"d0": carrier_freq})
#     signals = converter.get_signals(p)
#     fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))
#     for ax, title in zip(axs, ["envelope", "signal"]):
#         signals[0].draw(0, t_tot, 2000, title, axis=ax)
#         ax.set_xlabel("Time (ns)")
#         ax.set_ylabel("Amplitude")
#         ax.set_title(title)


# def qobj_to_Array(matrix):
#     if type(matrix) == qutip.qobj.Qobj:
#         return Array(matrix.full()) 
#     else:
#         return Array(matrix) # Should be able to directly convert to Array
    
