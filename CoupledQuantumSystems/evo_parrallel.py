from utils_v2_for_gfIFQ.utils_coupled_systems import CoupledSystem
import concurrent
from loky import get_reusable_executor
from typing import List, Any
from utils_v2_for_gfIFQ.utils_basic_funcs import *
from utils_v2_for_gfIFQ.utils_DriveTerm import *
from utils_v2_for_gfIFQ.utils_evo import ODEsolve_and_post_process   

def run_parallel_ODEsolve_and_post_process_jobs_with_different_systems(
        list_of_systems: List[CoupledSystem],
        list_of_kwargs: list[Any],
        max_workers = None,
        store_states = True,
        post_processing = ['pad_back'],
    ):
    '''
    This function helps run mesolve using the ODEsolve_and_post_process function concurrently
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

                static_hamiltonian=list_of_systems[i].diag_dressed_hamiltonian,
                drive_terms=list_of_kwargs[i].get('drive_terms', None),
                c_ops=list_of_kwargs[i].get('c_ops', None),
                e_ops=list_of_kwargs[i].get('e_ops', None),
                store_states = store_states,
                post_processing_funcs=post_processing_funcs,
                post_processing_args=post_processing_args,
                file_name = f'{i}')
            futures[future] = i
        
        for future in concurrent.futures.as_completed(futures):
            original_index = futures[future]
            results[original_index] = future.result()
    return results

def run_dq_ODEsolve_and_post_process_jobs_with_different_systems_but_same_y0(
        list_of_systems: List[CoupledSystem],
        list_of_kwargs: list[Any],
        max_workers = None,
        store_states = True,
        post_processing = ['pad_back'],
    ):
    '''
    This function runs dynamiqs's mesolve or sesolve using the concurrency of dynamiqs, 
       and then use cpu parrallization to do post processing.
    Function signature is intentionally kept the same as run_parallel_ODEsolve_and_post_process_jobs_with_different_systems

    Because we utilize dynamiqs' concurrency, we will run every hamiltonian with every initial states when calling dq.sesolve. 
    And most importantly we will apply the jump_ops to every simulation! We don't have the same jump_ops with different systems.
    '''
    pass

