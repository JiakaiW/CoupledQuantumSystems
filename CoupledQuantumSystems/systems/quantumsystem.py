import concurrent
from loky import get_reusable_executor
from typing import List, Union, Any
from ..dynamics import DriveTerm
from ..dynamics import ODEsolve_and_post_process
import numpy as np
import qutip

class QuantumSystem:
    """Base class for quantum systems providing common simulation functionality.

    This class serves as the foundation for specific quantum system implementations,
    offering methods for parallel quantum system evolution and state processing.
    It supports both master equation and Schrödinger equation evolution using
    QuTiP's solvers.

    Attributes:
        diag_hamiltonian (qutip.Qobj): Diagonalized Hamiltonian of the system.
        products_to_keep (List[List[int]]): List of product states to keep in the
            truncated Hilbert space.
        product_to_dressed (dict): Mapping from product states to dressed states.
        qbt_position (int): Position of the qubit in the product state indices.
    """

    def run_qutip_mesolve_parrallel(self,
                                    initial_states: Union[qutip.Qobj,
                                                         np.ndarray[qutip.Qobj]],
                                    tlist: Union[np.array, 
                                                 List[np.array]],
                                    drive_terms: Union[List[DriveTerm],
                                                       List[List[DriveTerm]]],
                                    c_ops: Union[None,
                                                 List[qutip.Qobj],
                                                 List[List[qutip.Qobj]]]=[],
                                    e_ops: Union[None,
                                                 List[qutip.Qobj],
                                                 List[List[qutip.Qobj]]]=[],
                                    show_each_thread_progress=False,
                                    show_multithread_progress=False,
                                    **kwargs
                                    ) -> Union[List[Any],
                                                List[List[Any]]]:
        """
        num_init_states = len(initial_states)  
        num_hamiltonian = 
          = shape(tlist)[0] if tlist is 2d
          = len(drive_terms) if drive_terms is List[List[DriveTerm]]
        if c_ops is a 1d list of qutip.Qobj, 
        or if e_ops is a 1d list of qutip.Qobj, 
        or if tlist is a single np.array,
        or if drive_terms is a 1d list of DriveTerm,
            then we broadcast them to 2d list (repeat the same operator for all Hamiltonian), as long as one of these inputs indicate multiple Hamiltonians
        if num_hamiltonian == 1:
            return a 1d list of results
        else:
            return a 2d list of results, each containing num_init_states evolution results 
            
        Returns:
            Union[List[Any], List[List[Any]]]: List of evolution results, possibly
                post-processed. The structure matches the input structure of
                initial_states and drive_terms.

        Example:
            >>> system = QuantumSystem()
            >>> initial_state = qutip.basis(2, 0)
            >>> tlist = np.linspace(0, 10, 100)
            >>> drive = DriveTerm(qutip.sigmax(), lambda t, args: np.sin(t))
            >>> results = system.run_qutip_mesolve_parrallel(
            ...     initial_states=[initial_state],
            ...     tlist=tlist,
            ...     drive_terms=[[drive]],
            ...     e_ops=[qutip.sigmaz()]
            ... )
        """
        if isinstance(initial_states, qutip.Qobj):
            num_init_states = 1
            initial_states = [initial_states]
        else:
            num_init_states = len(initial_states)

        # Now let's check how many Hamiltonians these different inputs indicate
        num_hamiltonian = 1
        num_hamiltonian_tlist = 1
        num_hamiltonian_drive_terms = 1
        num_hamiltonian_c_ops = 1
        num_hamiltonian_e_ops = 1

        if isinstance(tlist, np.ndarray): # only one type of evolution
            num_hamiltonian_tlist = 1
        else:
            num_hamiltonian_tlist = len(tlist)
        if isinstance(drive_terms[0], DriveTerm):
            num_hamiltonian_drive_terms = 1
        else:
            num_hamiltonian_drive_terms = len(drive_terms)
        if c_ops is None or len(c_ops)==0 or isinstance(c_ops[0], qutip.Qobj): # The order matters. Otherwise 'NoneType' object is not subscriptable
            num_hamiltonian_c_ops = 1
        else:
            num_hamiltonian_c_ops = len(c_ops)
        if e_ops is None or len(e_ops)==0 or isinstance(e_ops[0], qutip.Qobj):
            num_hamiltonian_e_ops = 1
        else:
            num_hamiltonian_e_ops = len(e_ops)


        if 1 == num_hamiltonian_tlist == num_hamiltonian_drive_terms == num_hamiltonian_c_ops == num_hamiltonian_e_ops:
            num_hamiltonian = 1
            tlist = [tlist for _ in range(num_init_states)]
            drive_terms = [drive_terms for _ in range(num_init_states)]
            c_ops = [c_ops for _ in range(num_init_states)]
            e_ops = [e_ops for _ in range(num_init_states)]
        else:
            num_hamiltonian = np.max([num_hamiltonian_tlist, num_hamiltonian_drive_terms, num_hamiltonian_c_ops, num_hamiltonian_e_ops])

            # Now broadcast the inputs to 2d list if needed
            if num_hamiltonian_tlist == 1:
                tlist = [tlist for _ in range(num_hamiltonian)]

            if num_hamiltonian_drive_terms == 1:
                drive_terms = [drive_terms for _ in range(num_hamiltonian)]
            else:
                # Check if drive_terms is a 2D list of DriveTerm
                assert isinstance(drive_terms, list) and all(isinstance(sublist, list) and all(isinstance(term, DriveTerm) for term in sublist) for sublist in drive_terms)

            if num_hamiltonian_c_ops == 1:
                c_ops = [c_ops for _ in range(num_hamiltonian)]
            else:
                assert isinstance(c_ops, list) and all(isinstance(sublist, list) and all(isinstance(op, qutip.Qobj) for op in sublist) for sublist in c_ops)

            if num_hamiltonian_e_ops == 1:
                e_ops = [e_ops for _ in range(num_hamiltonian)]
            else:
                assert isinstance(e_ops, list) and all(isinstance(sublist, list) and all(isinstance(op, qutip.Qobj) for op in sublist) for sublist in e_ops)

            assert len(drive_terms) == len(c_ops) == len(e_ops) == num_hamiltonian

        static_hamiltonian = self.diag_hamiltonian
        
        with get_reusable_executor(max_workers=None, context='loky') as executor:
            # Initialize results matrix
            results = [[None for _ in range(num_init_states)] for _ in range(num_hamiltonian)]
            
            # Submit all jobs and collect results as they complete
            futures = {}
            for i in range(num_hamiltonian):  # for each type of evolution
                for j in range(num_init_states):  # for each initial state
                    future = executor.submit(
                        ODEsolve_and_post_process,
                        y0=initial_states[j],
                        tlist=tlist[i],
                        static_hamiltonian=static_hamiltonian,
                        drive_terms=drive_terms[i],
                        c_ops=c_ops[i],
                        e_ops=e_ops[i],
                        print_progress=show_each_thread_progress,
                        **kwargs
                    )
                    futures[future] = (i, j)  # store both indices

            # Process results as they complete
            if show_multithread_progress:
                from tqdm import tqdm
                with tqdm(total=num_init_states*num_hamiltonian, desc="Processing simulations") as pbar:
                    for future in concurrent.futures.as_completed(futures):
                        i, j = futures[future]
                        results[i][j] = future.result()
                        pbar.update(1)
                        # Remove the completed future to free memory
                        del futures[future]
            else:
                for future in concurrent.futures.as_completed(futures):
                    i, j = futures[future]
                    results[i][j] = future.result()
                    # Remove the completed future to free memory
                    del futures[future]

        if num_hamiltonian == 1:
            return results[0]
        else:
            return results