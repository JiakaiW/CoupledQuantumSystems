import concurrent
from typing import List, Any
from loky import get_reusable_executor
from CoupledQuantumSystems.systems import QuantumSystem
from CoupledQuantumSystems.drive import *
from CoupledQuantumSystems.qobj_manip import pad_back_custom,dressed_to_2_level_dm
from CoupledQuantumSystems.evo import ODEsolve_and_post_process
from tqdm import tqdm

def run_parallel_ODEsolve_and_post_process_jobs_with_different_systems(
        list_of_systems: List[QuantumSystem],
        list_of_kwargs: list[Any],
        max_workers = None,
        store_states = True,
        post_processing = ['pad_back'],
    ):
    '''
    This function helps run qutip.mesolve using the ODEsolve_and_post_process function concurrently
    Args:
        list_of_systems: list of QuantumSystem
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
