import numpy as np
import qutip
from typing import List, Union, Any
from tqdm import tqdm
import concurrent
from loky import get_reusable_executor
from CoupledQuantumSystems.systems import CoupledSystem
from CoupledQuantumSystems.drive import *
from CoupledQuantumSystems.qobj_manip import pad_back_custom,dressed_to_2_level_dm
from qiskit.quantum_info import Operator
from qiskit_dynamics import Solver, Signal, SignalList

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

            rotating_frame: Union[bool, qutip.Qobj] = False,
            rwa_cutoff_freq: float = None,
            rwa_carrier_freq: List[float] = None,


            print_progress:bool = True,
            file_name: str = None,
            mcsolve_ntraj:int = 500,
            ):
    if (rotating_frame != False or rwa_cutoff_freq != None or rwa_carrier_freq != None) and method != 'qiskit_dynamics':
        raise ValueError("rotating_frame, rwa_cutoff_freq, and rwa_carrier_freq are only supported for qiskit_dynamics")
    
    if method in ['qutip.mesolve', 'qutip.mcsolve']:
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
                                ntraj = mcsolve_ntraj,
                                options=qutip.Options(store_states=True,num_cpus = None),
                                progress_bar = qutip.ui.progressbar.EnhancedTextProgressBar() if print_progress else None,
                                )
    elif method == 'qiskit_dynamics':
        qiskit_solver = Solver(
            static_hamiltonian=static_hamiltonian.full(),
            hamiltonian_operators=[drive_term.driven_op.full() for drive_term in drive_terms],
            rotating_frame=rotating_frame,
            rwa_cutoff_freq=rwa_cutoff_freq,
            rwa_carrier_freq=rwa_carrier_freq
        )
        results_qiskit_dynamics = qiskit_solver.solve(t_span=[tlist[0], tlist[-1]],
                                                      t_eval = tlist, 
                                                      y0=y0.full(), 
                                                      signals=[drive_term.envelope_to_qiskit_Signal() for drive_term in drive_terms], 
                                                      atol=1e-10, 
                                                      rtol=1e-10)
        result = qutip.solver.Result()
        result.times = tlist
        result.states = [qutip.Qobj(state) for state in results_qiskit_dynamics.y]
        if e_ops is not None:
            result.expect = np.array([
                [qutip.expect(e_op, state) for state in result.states] for e_op in e_ops
            ])
        else:
            result.expect = None
        result.solver = 'qiskit_dynamics'

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



def run_parallel_ODEsolve_and_post_process_jobs_with_different_systems(
        list_of_systems: List[CoupledSystem],
        list_of_kwargs: list[Any],
        max_workers = None,
        store_states = True,
        post_processing = ['pad_back'],
    ):
    '''
    This function helps run qutip.mesolve using the ODEsolve_and_post_process function concurrently
    Args:
        list_of_systems: list of CoupledSystem
        list_of_kwargs: list of kwargs dictionaries used to call ODEsolve_and_post_process
            a single kwargs should be a dictionary like {'y0',
                                                        'tlist',

                                                        'drive_terms',
                                                        'c_ops',
                                                        'e_ops'
                                                        }
    '''
    assert len(list_of_systems) == len(list_of_kwargs)
    
    results = [None] * len(list_of_systems)
    with get_reusable_executor(max_workers=max_workers, context='loky') as executor:
        futures = {}
        for i in range(len(list_of_systems)):
            post_processing_funcs = []
            post_processing_args = []
            if 'pad_back' in post_processing:
                post_processing_funcs.append(pad_back_custom)
                post_processing_args.append((list_of_systems[i].products_to_keep, 
                                            list_of_systems[i].product_to_dressed))
            if 'partial_trace_computational_states' in post_processing:
                post_processing_funcs.append(dressed_to_2_level_dm)
                post_processing_args.append((
                                            list_of_systems[i].product_to_dressed,
                                            list_of_systems[i].qbt_position, 
                                            list_of_systems[i].filtered_product_to_dressed,
                                            list_of_systems[i].sign_multiplier,
                                            None
                                            ))
            future = executor.submit(
                ODEsolve_and_post_process, 
                y0=list_of_kwargs[i]['y0'], 
                tlist=list_of_kwargs[i]['tlist'], 

                static_hamiltonian=list_of_systems[i].diag_hamiltonian,
                drive_terms=list_of_kwargs[i].get('drive_terms', None),
                c_ops=list_of_kwargs[i].get('c_ops', None),
                e_ops=list_of_kwargs[i].get('e_ops', None),
                store_states = store_states,
                post_processing_funcs=post_processing_funcs,
                post_processing_args=post_processing_args,
                file_name = f'{i}'
                )
            futures[future] = i
        
        for future in concurrent.futures.as_completed(futures):
            original_index = futures[future]
            results[original_index] = future.result()
    return results

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
