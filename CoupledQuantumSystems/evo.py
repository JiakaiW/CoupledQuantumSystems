from datetime import datetime
import numpy as np
import qutip
from typing import List, Union
from tqdm import tqdm
from CoupledQuantumSystems.frame import RotatingFrame, static_rwa
from CoupledQuantumSystems.qobj_manip import *
from CoupledQuantumSystems.drive import *

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

def ODEsolve_and_post_process(
            y0: qutip.Qobj,
            tlist: np.array, 

            static_hamiltonian: qutip.Qobj,
            drive_terms: List[DriveTerm],
            c_ops: Union[None,List[qutip.Qobj]] = None,
            e_ops:Union[None,List[qutip.Qobj]] = None,

            store_states = True,
            method:str = 'qutip.mesolve',
            post_processing_funcs:List=[],
            post_processing_args:List=[],

            apply_rwa: bool = False,
            cutoff_freq: float = 1.0,

            print_progress:bool = True,
            file_name: str = None,
            mcsolve_ntraj:int = 500,
            ):
    '''
    This method is only used for qutip's cpu solvers. For dynamiqs solver call CoupledSystem.run_dq_mesolve_parrallel

    It should take in:
        a static hamiltonian, 
        a list of "drive terms", 
            then assemble the two into an H_with_drive
        a list of c_ops
    '''
    if apply_rwa:
        frame       = RotatingFrame.from_operator(static_hamiltonian)
        new_static  = static_rwa(frame, static_hamiltonian, cutoff_freq)
        drive_terms_RWA = rotating_wave_approximation(frame, drive_terms, cutoff_freq)

        H_with_drives = [new_static] + [
            [dt.driven_op, dt.pulse_shape_func_with_id] for dt in drive_terms_RWA
        ]

        additional_args = {}
        for drive_term in drive_terms_RWA:
            for key in drive_term.pulse_shape_args_with_id:
                if key in additional_args:
                    raise ValueError(f"Duplicate key found: {key}")
                else:
                    additional_args[key] = drive_term.pulse_shape_args_with_id[key]
    else:
        H_with_drives =  [static_hamiltonian] + \
            [[drive_term.driven_op, drive_term.pulse_shape_func_with_id] for drive_term in drive_terms]
    
        additional_args = {}
        for drive_term in drive_terms:
            for key in drive_term.pulse_shape_args_with_id:
                if key in additional_args:
                    raise ValueError(f"Duplicate key found: {key}")
                else:
                    additional_args[key] = drive_term.pulse_shape_args_with_id[key]

    if method == 'qutip.mesolve':
        result = qutip.mesolve(
            rho0=y0,
            H=H_with_drives,
            tlist=tlist,
            c_ops=c_ops,
            e_ops = e_ops,
            args=additional_args,
            options=qutip.Options(store_states=store_states, nsteps=200000, num_cpus=1),
            progress_bar = qutip.ui.progressbar.EnhancedTextProgressBar() if print_progress else None,
        )

    elif method == 'qutip.mcsolve':
        result = qutip.mcsolve(psi0=y0, 
                            H= H_with_drives,
                            tlist=tlist,
                            args = additional_args,
                            c_ops=c_ops,
                            e_ops = e_ops,
                            ntraj = 500,
                            options=qutip.Options(store_states=True,num_cpus = None),
                            progress_bar = qutip.ui.progressbar.EnhancedTextProgressBar() if print_progress else None,
                            )
    else:
        raise Exception("solver method not supported")


    result = post_process(result,
                                 post_processing_funcs,
                                post_processing_args)
    # if file_name!= None:
    #     current_datetime = datetime.now()
    #     datetime_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    #     with open(f'{file_name} {datetime_string}.pkl', 'wb') as file:
    #         pickle.dump(result, file)
    return result
