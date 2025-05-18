import numpy as np
import qutip
from typing import List, Union, Any
from CoupledQuantumSystems.drive import *
from qiskit_dynamics import Solver

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
            static_dissipators= [op.full() for op in c_ops] if  c_ops is not None else None,
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

