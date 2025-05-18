from tqdm import tqdm
import concurrent
from loky import get_reusable_executor
from typing import List, Union, Any
from CoupledQuantumSystems.drive import DriveTerm
from CoupledQuantumSystems.evo import ODEsolve_and_post_process
from CoupledQuantumSystems.qobj_manip import generate_single_mapping,truncate_custom,pad_back_custom,dressed_to_2_level_dm, get_product_vectorized
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
                                    post_processing_funcs=[],
                                    post_processing_args=[],
                                    show_each_thread_progress=False,
                                    show_multithread_progress=False,
                                    store_states=True,
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
            
        Args:
            initial_states (Union[qutip.Qobj, np.ndarray[qutip.Qobj]]): Initial
                quantum states. Can be a single state or an array of states.
            tlist (Union[np.array, List[np.array]]): Time points for evolution.
                Can be a single array or a list of arrays.
            drive_terms (Union[List[DriveTerm], List[List[DriveTerm]]]): Drive
                terms for the Hamiltonian. Can be a single list or a list of lists.
            c_ops (Union[None, List[qutip.Qobj], List[List[qutip.Qobj]]]): Collapse
                operators for the master equation. Can be None, a single list, or
                a list of lists.
            e_ops (Union[None, List[qutip.Qobj], List[List[qutip.Qobj]]]): Expectation
                operators to calculate. Can be None, a single list, or a list of lists.
            post_processing_funcs (list, optional): List of functions to apply to
                results after evolution. Defaults to empty list.
            post_processing_args (list, optional): Arguments for post-processing
                functions. Defaults to empty list.
            show_each_thread_progress (bool, optional): Whether to show progress
                for each parallel thread. Defaults to False.
            show_multithread_progress (bool, optional): Whether to show overall
                parallel progress. Defaults to False.
            store_states (bool, optional): Whether to store quantum states during
                evolution. Defaults to True.

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
                        post_processing_funcs=post_processing_funcs,
                        post_processing_args=post_processing_args,
                        print_progress=show_each_thread_progress,
                        store_states = store_states
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
        

class CoupledSystem(QuantumSystem):
    """Base class for coupled quantum systems.

    This class extends QuantumSystem to handle coupled quantum systems, providing
    methods for state truncation, padding, and conversion between product and
    dressed state bases. It serves as the foundation for specific coupled system
    implementations like qubit-resonator systems.

    Attributes:
        hilbertspace (scqubits.HilbertSpace): Hilbert space of the coupled system.
        products_to_keep (List[List[int]]): List of product states to keep in the
            truncated Hilbert space.
        qbt_position (int): Position of the qubit in the product state indices.
        computaional_states (str): String specifying computational states, e.g., '0,1'.
        sign_multiplier (np.ndarray): Array of sign multipliers for state conversion.
    """

    def __init__(self,
                 hilbertspace,
                 products_to_keep,
                 qbt_position,
                 computaional_states):
        """Initialize a coupled quantum system.

        Args:
            hilbertspace (scqubits.HilbertSpace): Hilbert space of the coupled system.
            products_to_keep (List[List[int]]): List of product states to keep in the
                truncated Hilbert space.
            qbt_position (int): Position of the qubit in the product state indices.
            computaional_states (str): String specifying computational states, e.g., '0,1'.

        Example:
            >>> hilbertspace = scqubits.HilbertSpace([qubit, resonator])
            >>> system = CoupledSystem(
            ...     hilbertspace=hilbertspace,
            ...     products_to_keep=[[0,0], [0,1], [1,0], [1,1]],
            ...     qbt_position=0,
            ...     computaional_states='0,1'
            ... )
        """
        self.qbt_position = qbt_position
        self.computaional_states = computaional_states
        self.hilbertspace = hilbertspace
        if not self.hilbertspace.lookup_exists():
            self.hilbertspace.generate_lookup()
        self.evals = hilbertspace["evals"][0]
        self.evecs = hilbertspace["evecs"][0]
        self.product_to_dressed, failed = generate_single_mapping(
            self.hilbertspace.hamiltonian(), evals=self.evals, evecs=self.evecs)
        if failed:
            if hasattr(self, 'alternative_product_to_dressed') and callable(getattr(self, 'alternative_product_to_dressed')):
                self.product_to_dressed = self.alternative_product_to_dressed()

        self.set_new_product_to_keep(products_to_keep)

        self.set_sign_multiplier()

    def set_sign_multiplier(self):
        """Set sign multipliers for state conversion.

        This method sets up sign multipliers needed for converting between product
        and dressed state bases. It filters the product_to_dressed mapping to only
        include states relevant to the computational states and modifies qubit indices.

        Note:
            This method is called internally during system initialization and when
            the product states to keep are updated.
        """
        self.filtered_product_to_dressed = {}
        for product_state, dressed_index in self.product_to_dressed.items():
            if product_state[self.qbt_position] in (self.computaional_states[0], self.computaional_states[1]):
                new_product_state = list(product_state)
                new_product_state[self.qbt_position] = 0 if product_state[self.qbt_position] == self.computaional_states[0] else 1
                self.filtered_product_to_dressed[tuple(
                    new_product_state)] = dressed_index

        dressed_idxes_with_negative_sign = []
        for i in range(self.hilbertspace.dimension):
            arr = self.evecs[i].full()
            max_abs_index = np.argmax(np.abs(arr))
            max_abs_value = arr[max_abs_index]
            if max_abs_value > 0:
                pass
            elif max_abs_value < 0:
                dressed_idxes_with_negative_sign.append(i)

        # Convert dressed_idxes_with_negative_sign to a set for O(1) lookup
        dressed_idxes_with_negative_sign_set = set(
            dressed_idxes_with_negative_sign)

        # Pre-compute the sign multiplier for each dressed index
        self.sign_multiplier = {idx: -1 if idx in dressed_idxes_with_negative_sign_set else 1
                                for idx in self.product_to_dressed.values()}


        max_index = len(self.evals)# max(self.sign_multiplier.keys())
        self.sign_multiplier_vector = np.zeros(max_index + 1, dtype=int)
        for index, sign in self.sign_multiplier.items():
            self.sign_multiplier_vector[index] = sign

    def set_new_product_to_keep(self, products_to_keep):
        """Update the list of product states to keep.

        Args:
            products_to_keep (List[List[int]]): New list of product states to keep.

        Example:
            >>> system.set_new_product_to_keep([[0,0], [0,1], [1,0], [1,1], [2,0]])
        """
        if products_to_keep == None or products_to_keep == []:
            products_to_keep = list(
                product(*[range(dim) for dim in self.hilbertspace.subsystem_dims]))

        self.products_to_keep = products_to_keep
        self.diag_hamiltonian = self.truncate_function(qutip.Qobj((
            2 * np.pi * qutip.Qobj(np.diag(self.evals),
                                   dims=[self.hilbertspace.subsystem_dims] * 2)
        )[:, :]))

    def truncate_function(self, qobj):
        """Truncate a quantum object to the kept product states.

        Args:
            qobj (qutip.Qobj): Quantum object to truncate.

        Returns:
            qutip.Qobj: Truncated quantum object.

        Example:
            >>> truncated_state = system.truncate_function(state)
        """
        return truncate_custom(qobj, self.products_to_keep, self.product_to_dressed)

    def pad_back_function(self, qobj):
        """Pad a truncated quantum object back to full dimension.

        Args:
            qobj (qutip.Qobj): Truncated quantum object to pad.

        Returns:
            qutip.Qobj: Padded quantum object.

        Example:
            >>> padded_state = system.pad_back_function(truncated_state)
        """
        return pad_back_custom(qobj, self.products_to_keep, self.product_to_dressed)

    def convert_dressed_to_product_vectorized(self,
                                             states,
                                             products_to_keep,
                                             num_processes=None,
                                             update_products_to_keep=True,
                                             show_progress=False):
        """Convert dressed states to product states using vectorized operations.

        Args:
            states (np.ndarray): Array of dressed states to convert.
            products_to_keep (List[List[int]]): List of product states to keep.
            num_processes (int, optional): Number of processes to use for parallel
                computation. Defaults to None (use all available cores).
            update_products_to_keep (bool, optional): Whether to update the internal
                products_to_keep list. Defaults to True.
            show_progress (bool, optional): Whether to show progress bar.
                Defaults to False.

        Returns:
            np.ndarray: Array of converted product states.

        Example:
            >>> product_states = system.convert_dressed_to_product_vectorized(
            ...     dressed_states,
            ...     products_to_keep=[[0,0], [0,1], [1,0], [1,1]],
            ...     show_progress=True
            ... )
        """
        if update_products_to_keep:
            self.set_new_product_to_keep(products_to_keep)
            self.set_new_operators_after_setting_new_product_to_keep()
        
        if num_processes is None:
            num_processes = multiprocessing.cpu_count()
        
        ## non-vectorized multi processing
        # partial_function = partial(get_product,
        #                         pad_back_custom = self.pad_back_function,
        #                         product_to_dressed = self.product_to_dressed,
        #                         sign_multiplier = self.sign_multiplier)

        # with multiprocessing.Pool(processes=num_processes) as pool:
        #     product_states = pool.map(partial_function,states)

        # None-multiprocessing, vectorized
        # product_states  = []
        # for state in tqdm(states):
        #     product_states.append(get_product_vectorized(state,
        #                                                  self.pad_back_function,
        #                                                  self.product_to_dressed,
        #                                                  self.sign_multiplier_vector))

        # Multiprocessing, vectorized
        partial_function = partial(get_product_vectorized,
                        pad_back_custom = self.pad_back_function,
                        product_to_dressed = self.product_to_dressed,
                        sign_multiplier_vector = self.sign_multiplier_vector)

        with multiprocessing.Pool(processes=num_processes) as pool:
            if show_progress:
                product_states = list(tqdm(pool.imap(partial_function, states), total=len(states), desc="Processing States"))
            else:
                product_states = pool.map(partial_function, states)

        return product_states

    def run_qutip_mesolve_parrallel(self,
                                    initial_states: qutip.Qobj,  # truncated initial states
                                    tlist: np.array,
                                    drive_terms: Union[List[DriveTerm],
                                                       List[List[DriveTerm]]],
                                    c_ops: Union[None,
                                                 List[qutip.Qobj]] = None,
                                    e_ops: Union[None,
                                                 List[qutip.Qobj]] = None,

                                    post_processing=['pad_back'],
                                    show_each_thread_progress = True, 
                                    show_multithread_progress = False,
                                    store_states = True,
                                    ):
        post_processing_funcs = []
        post_processing_args = []
        if 'pad_back' in post_processing:
            post_processing_funcs.append(pad_back_custom)
            post_processing_args.append((self.products_to_keep,
                                         self.product_to_dressed))
        if 'partial_trace_computational_states' in post_processing:
            post_processing_funcs.append(dressed_to_2_level_dm)
            post_processing_args.append((
                                        self.product_to_dressed,
                                        self.qbt_position,
                                        self.filtered_product_to_dressed,
                                        self.sign_multiplier,
                                        None
                                        ))
        return super().run_qutip_mesolve_parrallel(initial_states=initial_states, 
                                                   tlist=tlist, 
                                                   drive_terms=drive_terms, 
                                                   c_ops=c_ops, 
                                                   e_ops=e_ops, 
                                                   post_processing_funcs=post_processing_funcs, 
                                                   post_processing_args=post_processing_args, 
                                                   show_each_thread_progress=show_each_thread_progress, 
                                                   show_multithread_progress=show_multithread_progress,
                                                   store_states=store_states)

class QubitResonatorSystem(CoupledSystem):
    """System consisting of a qubit coupled to a resonator.

    This class implements a coupled system where a qubit (e.g., transmon or fluxonium)
    is coupled to a resonator. It provides methods for setting up and manipulating
    the system's operators and parameters.

    Attributes:
        kappa (float): Decay rate of the resonator.
        g_strength (float): Coupling strength between qubit and resonator.
    """

    def set_new_operators_after_setting_new_product_to_keep(self):
        """Update system operators after changing the product states to keep.

        This method should be called after modifying the product states to keep
        to ensure all system operators are properly updated for the new basis.

        Note:
            This method is called internally when the product states to keep are
            updated through set_new_product_to_keep.
        """
        self.a_trunc = self.truncate_function(self.a)
        # self.truncate_function(self.hilbertspace.op_in_dressed_eigenbasis(self.osc.n_operator))
        self.driven_operator = self.a_trunc + self.a_trunc.dag()
        self.set_new_kappa(self.kappa)

    def set_new_kappa(self, kappa):
        """Update the resonator decay rate.

        Args:
            kappa (float): New decay rate for the resonator.

        Example:
            >>> system.set_new_kappa(0.01)  # Set resonator decay rate to 0.01
        """
        self.kappa = kappa
        self.c_ops = [np.sqrt(self.kappa) * self.a_trunc]

    def get_ladder_overlap_arr(self, resonator_creation_arr):
        """Calculate overlap between resonator ladder operators and dressed states.

        Args:
            resonator_creation_arr (np.ndarray): Array representing the resonator
                creation operator in the product basis.

        Returns:
            np.ndarray: Array of overlaps between the resonator ladder operator
                and the dressed states.

        Note:
            The input resonator_creation_arr should be an identity-padded array
            where each row represents a dressed ket in the product basis.
        """
        # resonator_creation_arr should be an id padded arr
        # each row (first index) is a dressed ket in product basis
        evecs_arr_row_dressed_ket = scqubits.utils.spectrum_utils.convert_evecs_to_ndarray(self.evecs)
        # now each column (second index) is a dressed ket in product basis
        evecs_arr_column_dressed_ket = evecs_arr_row_dressed_ket.T
        # each column is a dressed state after creation in product basis
        evecs_after_creation_arr_column_dressed_ket = resonator_creation_arr @ evecs_arr_column_dressed_ket
        denominator_1d = np.sum(np.abs(evecs_after_creation_arr_column_dressed_ket) ** 2, axis=0)
        # (row i, column j) is the overlap between i and creation@j
        numerator = evecs_arr_row_dressed_ket.conj() @ evecs_after_creation_arr_column_dressed_ket
        # (row i, column j) is the normalized overlap between i and creation@j
        ladder_overlap = (numerator/denominator_1d)**2
        ladder_overlap = np.abs(ladder_overlap)
        return ladder_overlap
    
class FluxoniumOscillatorSystem(QubitResonatorSystem):
    """System consisting of a fluxonium qubit coupled to an oscillator.

    This class implements a coupled system where a fluxonium qubit is coupled to
    an oscillator (resonator). It provides methods for setting up and manipulating
    the system's parameters and operators.

    Attributes:
        EJ (float): Josephson energy of the fluxonium.
        EC (float): Charging energy of the fluxonium.
        EL (float): Inductive energy of the fluxonium.
        qubit_level (float): Number of levels to keep for the fluxonium.
        Er (float): Energy of the resonator.
        osc_level (float): Number of levels to keep for the oscillator.
        kappa (float): Decay rate of the resonator.
        g_strength (float): Coupling strength between fluxonium and resonator.
    """

    def __init__(self,
                 computaional_states: str = '1,2',
                 EJ: float = None,
                 EC: float = None,
                 EL: float = None,
                 qubit_level: float = 13,
                 qbt: scqubits.Fluxonium = None,
                 Er: float = None,
                 osc_level: float = 30,
                 osc: scqubits.Oscillator = None,
                 kappa=0.001,
                 g_strength: float = None,
                 products_to_keep: List[List[int]] = None):
        """Initialize a fluxonium-oscillator coupled system.

        Args:
            computaional_states (str, optional): String specifying computational states.
                Defaults to '1,2'.
            EJ (float, optional): Josephson energy of the fluxonium. Defaults to None.
            EC (float, optional): Charging energy of the fluxonium. Defaults to None.
            EL (float, optional): Inductive energy of the fluxonium. Defaults to None.
            qubit_level (float, optional): Number of levels to keep for the fluxonium.
                Defaults to 13.
            qbt (scqubits.Fluxonium, optional): Pre-initialized fluxonium object.
                Defaults to None.
            Er (float, optional): Energy of the resonator. Defaults to None.
            osc_level (float, optional): Number of levels to keep for the oscillator.
                Defaults to 30.
            osc (scqubits.Oscillator, optional): Pre-initialized oscillator object.
                Defaults to None.
            kappa (float, optional): Decay rate of the resonator. Defaults to 0.001.
            g_strength (float, optional): Coupling strength between fluxonium and
                resonator. Defaults to None.
            products_to_keep (List[List[int]], optional): List of product states to
                keep. Defaults to None.

        Example:
            >>> system = FluxoniumOscillatorSystem(
            ...     computaional_states='1,2',
            ...     EJ=2.33,
            ...     EC=0.69,
            ...     EL=0.12,
            ...     qubit_level=13,
            ...     Er=7.165,
            ...     osc_level=30,
            ...     kappa=0.001,
            ...     g_strength=0.18
            ... )
        """
        if qbt is not None:
            self.qbt = qbt
        else:
            self.qbt = scqubits.Fluxonium(EJ=EJ, EC=EC, EL=EL, flux=0, cutoff=110, truncated_dim=qubit_level)
        
        if osc is not None:
            self.osc = osc
        else:
            # l_osc should have been 1/sqrt(2), otherwise I'm effectively reducing the coupling strength by sqrt(2)
            self.osc = scqubits.Oscillator(E_osc=Er, truncated_dim=osc_level, l_osc=1.0)

        # https://scqubits.readthedocs.io/en/latest/api-doc/_autosummary/scqubits.core.oscillator.Oscillator.html#scqubits.core.oscillator.Oscillator.n_operator
        hilbertspace = scqubits.HilbertSpace([self.qbt, self.osc])
        hilbertspace.add_interaction(
            g_strength=g_strength, op1=self.qbt.n_operator, op2=self.osc.n_operator, add_hc=False)  # Edited

        super().__init__(hilbertspace=hilbertspace,
                         products_to_keep=products_to_keep,
                         qbt_position=0,
                         computaional_states=[int(computaional_states[0]), int(computaional_states[-1])])

        self.a = qutip.Qobj(self.hilbertspace.op_in_dressed_eigenbasis(
            self.osc.annihilation_operator)[:, :])
        self.kappa = kappa
        self.set_new_operators_after_setting_new_product_to_keep()

class TransmonOscillatorSystem(QubitResonatorSystem):
    """System consisting of a transmon qubit coupled to an oscillator.

    This class implements a coupled system where a transmon qubit is coupled to
    an oscillator (resonator). It provides methods for setting up and manipulating
    the system's parameters and operators.

    Attributes:
        kappa (float): Decay rate of the resonator.
        g_strength (float): Coupling strength between transmon and resonator.
    """

    def __init__(self,
                 qbt: scqubits.Transmon = None,
                 osc: scqubits.Oscillator = None,
                 kappa=0.01,
                 g_strength: float = None,
                 products_to_keep: List[List[int]] = None):
        """Initialize a transmon-oscillator coupled system.

        Args:
            qbt (scqubits.Transmon, optional): Pre-initialized transmon object.
                Defaults to None.
            osc (scqubits.Oscillator, optional): Pre-initialized oscillator object.
                Defaults to None.
            kappa (float, optional): Decay rate of the resonator. Defaults to 0.01.
            g_strength (float, optional): Coupling strength between transmon and
                resonator. Defaults to None.
            products_to_keep (List[List[int]], optional): List of product states to
                keep. Defaults to None.

        Example:
            >>> system = TransmonOscillatorSystem(
            ...     qbt=transmon,
            ...     osc=oscillator,
            ...     kappa=0.01,
            ...     g_strength=0.1
            ... )
        """
        self.qbt = qbt        
        self.osc = osc

        # https://scqubits.readthedocs.io/en/latest/api-doc/_autosummary/scqubits.core.oscillator.Oscillator.html#scqubits.core.oscillator.Oscillator.n_operator
        hilbertspace = scqubits.HilbertSpace([self.qbt, self.osc])
        hilbertspace.add_interaction(
            g_strength=g_strength, op1=self.qbt.n_operator, op2=self.osc.n_operator, add_hc=False)  # Edited

        super().__init__(hilbertspace=hilbertspace,
                         products_to_keep=products_to_keep,
                         qbt_position=0,
                         computaional_states=[0,1])

        self.a = qutip.Qobj(self.hilbertspace.op_in_dressed_eigenbasis(
            self.osc.annihilation_operator)[:, :])
        self.kappa = kappa
        self.set_new_operators_after_setting_new_product_to_keep()

    def alternative_product_to_dressed(self) -> dict:
        """Generate an alternative mapping between product and dressed states.

        This method provides an alternative way to map between product states and
        dressed states, which can be useful in cases where the standard mapping
        fails or is not optimal.

        Returns:
            dict: Mapping from product states to dressed states.

        Note:
            This method is called internally when the standard mapping generation
            fails during system initialization.
        """
        id_wrapped_resonator_destory = qutip.tensor(qutip.identity(self.qbt.truncated_dim), qutip.destroy(self.osc.truncated_dim))
        resonator_creation_arr = id_wrapped_resonator_destory.dag().full()
        ladder_overlap = self.get_ladder_overlap_arr(resonator_creation_arr)
        overlap_idx_arr = np.zeros((self.qbt.truncated_dim,self.osc.truncated_dim),dtype=int)
        for ql in tqdm(range(self.qbt.truncated_dim), desc = "Using Blais approach to relable dressed states, ql loop:"):
            for ol in range(self.osc.truncated_dim):    
                if ql == 0 and ol == 0:
                    overlap_idx_arr[ql,ol] = 0
                elif ol == 0:
                    overlap_idx_arr[ql,ol] = self.product_to_dressed[(ql,ol)]
                    # print(f"overlap_idx_arr[{ql},0] = {overlap_idx_arr[ql,ol]}")
                else: #(ol > 0)
                    overlap_idx_arr[ql,ol] = np.argmax(ladder_overlap[:,overlap_idx_arr[ql,ol-1]])
        for ql in range(self.qbt.truncated_dim):
            for ol in range(self.osc.truncated_dim):
                self.product_to_dressed[(ql,ol)] =  overlap_idx_arr[ql,ol] 

        return self.product_to_dressed
    
class FluxoniumTransmonSystem(CoupledSystem):
    """System consisting of a fluxonium qubit coupled to a transmon qubit.

    This class implements a coupled system where a fluxonium qubit is coupled to
    a transmon qubit. It provides methods for setting up and manipulating the
    system's parameters and operators.

    Attributes:
        fluxonium (scqubits.Fluxonium): The fluxonium qubit object.
        transmon (scqubits.Transmon): The transmon qubit object.
        computaional_states (str): String specifying computational states.
        g_strength (float): Coupling strength between fluxonium and transmon.
    """

    def __init__(self,
                 fluxonium: scqubits.Fluxonium,
                 transmon: scqubits.Transmon,
                 computaional_states: str,  # = '0,1' or '1,2'
                 g_strength: float = 0.18,
                 products_to_keep: List[List[int]] = None,
                 ):
        """Initialize a fluxonium-transmon coupled system.

        Args:
            fluxonium (scqubits.Fluxonium): The fluxonium qubit object.
            transmon (scqubits.Transmon): The transmon qubit object.
            computaional_states (str): String specifying computational states,
                e.g., '0,1' or '1,2'.
            g_strength (float, optional): Coupling strength between fluxonium and
                transmon. Defaults to 0.18.
            products_to_keep (List[List[int]], optional): List of product states to
                keep. Defaults to None.

        Example:
            >>> system = FluxoniumTransmonSystem(
            ...     fluxonium=fluxonium_qubit,
            ...     transmon=transmon_qubit,
            ...     computaional_states='0,1',
            ...     g_strength=0.18
            ... )
        """
        self.fluxonium = fluxonium
        self.transmon = transmon
        hilbertspace = scqubits.HilbertSpace([self.fluxonium, self.transmon])
        hilbertspace.add_interaction(
            g_strength=g_strength, op1=self.fluxonium.n_operator, op2=self.transmon.n_operator, add_hc=False)

        super().__init__(hilbertspace=hilbertspace,
                         products_to_keep=products_to_keep,
                         qbt_position=0,
                         computaional_states=[int(computaional_states[0]), int(computaional_states[-1])])

class FFTSystem(CoupledSystem):
    """System consisting of two fluxonium qubits coupled to a transmon qubit.

    This class implements a coupled system where two fluxonium qubits are coupled
    to a transmon qubit, forming a three-qubit system. It provides methods for
    setting up and manipulating the system's parameters and operators, including
    special methods for implementing SATD (Shortcut to Adiabaticity) CZ gates.

    Attributes:
        fluxonium1 (scqubits.Fluxonium): First fluxonium qubit.
        fluxonium2 (scqubits.Fluxonium): Second fluxonium qubit.
        transmon (scqubits.Transmon): Transmon qubit.
        computaional_states (str): String specifying computational states.
        g_f1f2 (float): Coupling strength between fluxonium qubits.
        g_f1t (float): Coupling strength between first fluxonium and transmon.
        g_f2t (float): Coupling strength between second fluxonium and transmon.
    """

    def __init__(self,
                 fluxonium1: scqubits.Fluxonium,
                 fluxonium2: scqubits.Fluxonium,
                 transmon: scqubits.Transmon,
                 computaional_states: str,  # = '0,1' or '1,2'
                 g_f1f2: float = 0.1,
                 g_f1t: float = 0.1,
                 g_f2t: float = 0.1,
                 products_to_keep: List[List[int]] = None):
        """Initialize a two-fluxonium-one-transmon coupled system.

        Args:
            fluxonium1 (scqubits.Fluxonium): First fluxonium qubit.
            fluxonium2 (scqubits.Fluxonium): Second fluxonium qubit.
            transmon (scqubits.Transmon): Transmon qubit.
            computaional_states (str): String specifying computational states,
                e.g., '0,1' or '1,2'.
            g_f1f2 (float, optional): Coupling strength between fluxonium qubits.
                Defaults to 0.1.
            g_f1t (float, optional): Coupling strength between first fluxonium and
                transmon. Defaults to 0.1.
            g_f2t (float, optional): Coupling strength between second fluxonium and
                transmon. Defaults to 0.1.
            products_to_keep (List[List[int]], optional): List of product states to
                keep. Defaults to None.

        Example:
            >>> system = FFTSystem(
            ...     fluxonium1=fluxonium1,
            ...     fluxonium2=fluxonium2,
            ...     transmon=transmon,
            ...     computaional_states='0,1',
            ...     g_f1f2=0.1,
            ...     g_f1t=0.1,
            ...     g_f2t=0.1
            ... )
        """
        self.fluxonium1 = fluxonium1
        self.fluxonium2 = fluxonium2
        self.transmon = transmon
        hilbertspace = scqubits.HilbertSpace(
            [self.fluxonium1, self.fluxonium2, self.transmon])
        hilbertspace.add_interaction(
            g_strength=g_f1f2, op1=self.fluxonium1.n_operator, op2=self.fluxonium2.n_operator, add_hc=False)
        hilbertspace.add_interaction(
            g_strength=g_f1t, op1=self.fluxonium1.n_operator, op2=self.transmon.n_operator, add_hc=False)
        hilbertspace.add_interaction(
            g_strength=g_f2t, op1=self.fluxonium2.n_operator, op2=self.transmon.n_operator, add_hc=False)

        super().__init__(hilbertspace=hilbertspace,
                         products_to_keep=products_to_keep,
                         qbt_position=0,# let's use an arbitrary choice of qbt_position here.
                         computaional_states=[int(computaional_states[0]), int(computaional_states[-1])],# computaional_states here is also arbitraray, since it's only used to generate filtered_product_to_dressed to take out the computational subspace of the qubit in post-processing.
                         )
    
    def get_SATD_CZ_drive_terms(self):
        """Generate drive terms for implementing a SATD CZ gate.

        This method generates the necessary drive terms for implementing a
        Shortcut to Adiabaticity (SATD) CZ gate between the fluxonium qubits.

        Returns:
            List[DriveTerm]: List of drive terms for the SATD CZ gate.

        Note:
            The SATD CZ gate implementation follows the protocol described in
            the literature for fast and high-fidelity two-qubit gates.
        """
        def P(x):
            return 6*(2*x)**5-15*(2*x)**4+10*(2*x)**3
        def theta(t,tg):
            t_over_tg = t/tg
            if t_over_tg <= 1/2:
                return np.pi/2*P(t_over_tg)
            else:
                return np.pi/2*(1-P(t_over_tg - 1/2))
        def theta_t_gradient(t,tg):
            t_over_tg = t/tg
            # Derivative of P(x)
            def dP(x):
                return (960*x**4 - 960*x**3 + 240*x**2)/tg

            if t_over_tg <= 0.5:
                return (np.pi/2) * dP(t_over_tg)
            else:
                return (np.pi/2) * (-dP(t_over_tg - 0.5))
        def theta_t_curvature(t,tg):
            t_over_tg = t/tg
            # Second derivative of P(x)
            def ddP(x):
                return (3840*x**3 - 2880*x**2 + 480*x)/tg**2

            if t_over_tg <= 0.5:
                return (np.pi/2) * ddP(t_over_tg)
            else:
                return (np.pi/2) * (-ddP(t_over_tg - 0.5))

        def gamma(gamma_0,t_over_tg):
            if t_over_tg <= 1/2:
                return 0
            else:
                return gamma_0
        def Omega_A(t,tg,Omega_0):
            return Omega_0*np.sin(theta(t/tg))
        def Omega_B(t,tg,Omega_0,gamma_0):
            return Omega_0*np.cos(theta(t/tg))*np.exp(1j*gamma(gamma_0,t/tg))
        def Omega_A_tilde(t,tg,Omega_0):
            return Omega_0*()
        def Omega_B_tilde(t,tg,Omega_0,gamma_0):
            return Omega_0*np.sin(theta(t/tg))*np.exp(1j*gamma(gamma_0,t/tg))
        def SATD_coupler_modulation(t,args):
            pass
        drive_terms = [
            DriveTerm(
                driven_op=qutip.Qobj(
                    self.transmon.cos_phi_operator(energy_esys=True)),
                pulse_shape_func=SATD_coupler_modulation,
                pulse_id='SATD',  # Stoke is the first pulse, pump is the second
                pulse_shape_args={
                    'w_d': None,
                    
                },
            ),
            
        ]
        return drive_terms

class FluxoniumOscillatorFilterSystem(CoupledSystem):
    """System consisting of a fluxonium qubit coupled to an oscillator and a filter.

    This class implements a coupled system where a fluxonium qubit is coupled to
    an oscillator (resonator) and a filter oscillator. The filter is used to
    suppress unwanted Purcell decay while maintaining good readout properties.
    The system is based on the design described in Phys. Rev A 92, 012325 (2015).

    Attributes:
        qbt (scqubits.Fluxonium): The fluxonium qubit.
        osc (scqubits.Oscillator): The readout resonator.
        filter (scqubits.Oscillator): The filter oscillator.
        G_strength (float): Coupling strength between resonator and filter.
        a (qutip.Qobj): Annihilation operator for the resonator in dressed basis.
        a_trunc (qutip.Qobj): Truncated annihilation operator for the resonator.
        b (qutip.Qobj): Annihilation operator for the filter in dressed basis.
        b_trunc (qutip.Qobj): Truncated annihilation operator for the filter.
        driven_operator (qutip.Qobj): Operator used for driving the system.
        c_ops (List[qutip.Qobj]): Collapse operators for the system.
        w_d (float, optional): Drive frequency if specified.
    """

    def __init__(self,
                 computaional_states: str,  # = '0,1' or '1,2'
                 EJ: float = 2.33,
                 EC: float = 0.69,
                 EL: float = 0.12,
                 qubit_level: float = 13,
                 Er: float = 7.16518677,
                 osc_level: float = 20,
                 Ef: float = 7.13,
                 filter_level: float = 7,
                 kappa_f: float = 1.5,
                 g_strength: float = 0.18,
                 G_strength: float = 0.3,
                 products_to_keep: List[List[int]] = None,
                 w_d: float = None):
        """Initialize a fluxonium-oscillator-filter coupled system.

        Args:
            computaional_states (str): String specifying computational states,
                e.g., '0,1' or '1,2'.
            EJ (float, optional): Josephson energy of the fluxonium. Defaults to 2.33.
            EC (float, optional): Charging energy of the fluxonium. Defaults to 0.69.
            EL (float, optional): Inductive energy of the fluxonium. Defaults to 0.12.
            qubit_level (float, optional): Number of levels to keep for the fluxonium.
                Defaults to 13.
            Er (float, optional): Energy of the readout resonator. Defaults to 7.16518677.
            osc_level (float, optional): Number of levels to keep for the resonator.
                Defaults to 20.
            Ef (float, optional): Energy of the filter oscillator. Defaults to 7.13.
            filter_level (float, optional): Number of levels to keep for the filter.
                Defaults to 7.
            kappa_f (float, optional): Decay rate of the filter. Defaults to 1.5.
            g_strength (float, optional): Coupling strength between fluxonium and
                resonator. Defaults to 0.18.
            G_strength (float, optional): Coupling strength between resonator and
                filter. Defaults to 0.3.
            products_to_keep (List[List[int]], optional): List of product states to
                keep. Defaults to None.
            w_d (float, optional): Drive frequency. Defaults to None.

        Note:
            The filter is designed to suppress Purcell decay while maintaining
            good readout properties. The coupling strength G_strength is related
            to the resonator and filter energies and decay rates through the
            relation in Phys. Rev A 92, 012325 (2015).

        Example:
            >>> system = FluxoniumOscillatorFilterSystem(
            ...     computaional_states='0,1',
            ...     EJ=2.33,
            ...     EC=0.69,
            ...     EL=0.12,
            ...     qubit_level=13,
            ...     Er=7.165,
            ...     osc_level=20,
            ...     Ef=7.13,
            ...     filter_level=7,
            ...     kappa_f=1.5,
            ...     g_strength=0.18,
            ...     G_strength=0.3
            ... )
        """
        self.G_strength = G_strength

        self.qbt = scqubits.Fluxonium(
            EJ=EJ, EC=EC, EL=EL, flux=0, cutoff=110, truncated_dim=qubit_level)
        self.osc = scqubits.Oscillator(E_osc=Er, truncated_dim=osc_level)
        self.filter = scqubits.Oscillator(E_osc=Ef, truncated_dim=filter_level)
        hilbertspace = scqubits.HilbertSpace([self.qbt, self.osc, self.filter])
        hilbertspace.add_interaction(
            g_strength=g_strength, op1=self.qbt.n_operator, op2=self.osc.creation_operator, add_hc=True)
        hilbertspace.add_interaction(g_strength=G_strength, op1=self.osc.creation_operator,
                                     op2=self.filter.annihilation_operator, add_hc=True)

        super().__init__(hilbertspace=hilbertspace,
                         products_to_keep=products_to_keep,
                         qbt_position=0,
                         computaional_states=[int(computaional_states[0]), int(computaional_states[-1])])

        self.a = qutip.Qobj(self.hilbertspace.op_in_dressed_eigenbasis(
            self.osc.annihilation_operator)[:, :])
        self.a_trunc = self.truncate_function(self.a)

        self.b = qutip.Qobj(self.hilbertspace.op_in_dressed_eigenbasis(
            self.filter.annihilation_operator)[:, :])
        self.b_trunc = self.truncate_function(self.b)
        self.driven_operator = self.b_trunc+self.b_trunc.dag()
        self.c_ops = [np.sqrt(kappa_f) * self.b_trunc]

        if w_d != None:
            self.w_d = w_d
