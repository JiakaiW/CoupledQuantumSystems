import numpy as np
import qutip
from typing import List, Union, Any, Dict, Optional, Tuple
from .drive import *
from .drive_symbo import *
from qiskit_dynamics import Solver, Signal
from tqdm import tqdm
def ODEsolve_and_post_process(
            y0: qutip.Qobj,
            tlist: np.array, 

            static_hamiltonian: qutip.Qobj,
            drive_terms: List[DriveTerm],
            c_ops: Union[None,List[qutip.Qobj]] = None,
            e_ops:Union[None,List[qutip.Qobj]] = None,

            store_states = True,
            method:str = 'qutip.mesolve',
            qiskit_solver_method: Optional[str] = None,
            post_processing_funcs:List=[],
            post_processing_args:List=[],

            rotating_frame: Union[bool, qutip.Qobj] = False,
            rwa_cutoff_freq: float = None,
            rwa_carrier_freqs: List[float] = None,


            print_progress:bool = True,
            file_name: str = None,
            mcsolve_ntraj:int = 500,
            ):
    if (rotating_frame != False or rwa_cutoff_freq != None or rwa_carrier_freqs != None) and method != 'qiskit_dynamics':
        raise ValueError("rotating_frame, rwa_cutoff_freq, and rwa_carrier_freqs are only supported for qiskit_dynamics")
    
    if method in ['qutip.mesolve', 'qutip.mcsolve']:
        H =  [static_hamiltonian] 
        for drive_term in drive_terms:
            if isinstance(drive_term, DriveTerm):
                # Old DriveTerm type
                H.append([drive_term.driven_op, drive_term.pulse_shape_func_with_id])
            else:
                # New DriveTermSymbolic type
                H.append([drive_term.driven_op, drive_term.numpy_coeff])
        additional_args = {}
        for drive_term in drive_terms:
            if isinstance(drive_term, DriveTerm):
                # Old DriveTerm type
                for key in drive_term.pulse_shape_args_with_id:
                    if key in additional_args:
                        raise ValueError(f"Duplicate key found: {key}")
                    else:
                        additional_args[key] = drive_term.pulse_shape_args_with_id[key]
            else:
                # New DriveTermSymbolic type
                # No need to handle args as they're already substituted in __post_init__
                pass

        if method == 'qutip.mesolve':
            result = qutip.mesolve(
                rho0=y0,
                H=H,
                tlist=tlist,
                c_ops=c_ops,
                e_ops = e_ops,
                args=additional_args,
                options=qutip.Options(store_states=store_states, nsteps=200000, num_cpus=1),
                progress_bar = qutip.ui.progressbar.EnhancedTextProgressBar() if print_progress else None,
            )

        elif method == 'qutip.mcsolve':
            result = qutip.mcsolve(psi0=y0, 
                                H= H,
                                tlist=tlist,
                                args = additional_args,
                                c_ops=c_ops,
                                e_ops = e_ops,
                                ntraj = mcsolve_ntraj,
                                options=qutip.Options(store_states=True,num_cpus = None),
                                progress_bar = qutip.ui.progressbar.EnhancedTextProgressBar() if print_progress else None,
                                )
    elif method == 'qiskit_dynamics':
        qiskit_solver = Solver(
            static_hamiltonian=static_hamiltonian.full(),
            hamiltonian_operators=[drive_term.driven_op.full() for drive_term in drive_terms],
            static_dissipators= [op.full() for op in c_ops] if  c_ops is not None else None,
            rotating_frame=rotating_frame.full() if isinstance(rotating_frame, qutip.Qobj) else rotating_frame,
            rwa_cutoff_freq=rwa_cutoff_freq,
            rwa_carrier_freqs=rwa_carrier_freqs
        )
        signals = []
        for drive_term in drive_terms:
            if isinstance(drive_term, DriveTerm):
                # Old DriveTerm type
                signals.append(drive_term.envelope_to_qiskit_Signal())
            else:
                # New DriveTermSymbolic type
                if JAX_AVAILABLE:
                    if print_progress: print("JAX is available, using to_qiskit_signal_jax.") # More informative print
                    signals.append(drive_term.to_qiskit_signal_jax())
                else:
                    if print_progress: print("JAX is not available, using to_qiskit_signal_numpy.")
                    signals.append(drive_term.to_qiskit_signal_numpy())
        
        # Pass qiskit_solver_method to the solve call if provided
        solve_kwargs = {
            't_span': [tlist[0], tlist[-1]],
            't_eval': tlist,
            'y0': y0.full(),
            'signals': signals,
            'atol': 1e-10, # Default from before, can be parameterized
            'rtol': 1e-10  # Default from before, can be parameterized
        }
        if qiskit_solver_method:
            solve_kwargs['method'] = qiskit_solver_method
            
        results_qiskit_dynamics = qiskit_solver.solve(**solve_kwargs)
        results_qiskit_dynamics_y_array = np.array(results_qiskit_dynamics.y)
        # Convert results to QuTiP Result object format
        qobj_states = [qutip.Qobj(state) for state in results_qiskit_dynamics_y_array] # Let qutip figure out the dims automatically
        calculated_expect = []
        if e_ops:
            for op in e_ops:
                calculated_expect.append(np.array([qutip.expect(op, st) for st in qobj_states]))
        
        # num_collapse is 0 because this is not mcsolve
        # Create a minimal Result object first, then populate it to avoid constructor issues.
        result = qutip.solver.Result()
        result.states = qobj_states
        result.times = tlist
        result.expect = calculated_expect
        result.num_expect = len(e_ops) if e_ops else 0
        result.num_collapse = 0
        result.options = qutip.Options()
        result.solver = 'qiskit_dynamics (via ODEsolve_and_post_process)'

    elif method =='dynamiqs':
        raise NotImplementedError("dynamiqs is not implemented yet")
        pass
    else:
        raise Exception("solver method not supported")


    result = post_process(result,
                                post_processing_funcs,
                                post_processing_args)
        
    if file_name!= None:
        from datetime import datetime
        import pickle
        current_datetime = datetime.now()
        datetime_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        with open(f'{file_name} {datetime_string}.pkl', 'wb') as file:
            pickle.dump(result, file)
    return result

def post_process(
            result:qutip.solver.Result,
            post_processing_funcs:List=[],
            post_processing_args:List=[],
            show_progress:bool = False,
            ):
    # for func, args in zip(post_processing_funcs, post_processing_args):
    #     result.states = [func(state, *args) for state in tqdm(result.states, desc=f"Processing states with {func.__name__}")]

    # Editted post processing so that it doesn't overwrite result.states
    last_attribute_name = "states"
    for func, args in zip(post_processing_funcs, post_processing_args):
        new_attr_name = f"states_{func.__name__}" 
        if show_progress:
            processed_states = [func(state, *args) for state in tqdm( getattr(result, last_attribute_name), desc=f"Processing states with {func.__name__}")]
        else:
            processed_states = [func(state, *args) for state in getattr(result, last_attribute_name)]
        setattr(result, new_attr_name, processed_states)
        last_attribute_name = new_attr_name
    return result