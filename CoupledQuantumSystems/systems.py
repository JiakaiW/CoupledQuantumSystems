import concurrent
from itertools import product
import multiprocessing
from loky import get_reusable_executor
import numpy as np
import qutip
import scqubits
from typing import List, Union, Tuple
from functools import partial
from tqdm import tqdm

from CoupledQuantumSystems.qobj_manip import generate_single_mapping,truncate_custom,pad_back_custom,dressed_to_2_level_dm
from CoupledQuantumSystems.drive import DriveTerm
from CoupledQuantumSystems.evo import ODEsolve_and_post_process
from CoupledQuantumSystems.qobj_manip import get_product, get_product_vectorized

############################################################################
#
# Classes about modelling the system and running ODE solvers
#   the code is centered around qutip, for functions that use qiskit-dynamics,
#   I convert objects to jnp locally
#
############################################################################


class CoupledSystem:
    '''
    A parent class for quantum systems involving qubits and oscillators,

    This class is meant to be very generic, any specific setup can inherit from this 
        class and define commonly used attributes in the child class and be as customized as wanted
    '''

    def __init__(self,
                 hilbertspace,
                 products_to_keep,
                 qbt_position,
                 computaional_states):
        self.qbt_position = qbt_position
        self.computaional_states = computaional_states
        self.hilbertspace = hilbertspace
        self.hilbertspace.generate_lookup()
        self.evals = hilbertspace["evals"][0]
        self.evecs = hilbertspace["evecs"][0]
        self.product_to_dressed = generate_single_mapping(
            self.hilbertspace.hamiltonian(), evals=self.evals, evecs=self.evecs)

        self.set_new_product_to_keep(products_to_keep)

        self.set_sign_multiplier()

    def set_sign_multiplier(self):
        #############################################################################################
        #############################################################################################
        # TODO: This part about getting negative signs can be written more elegantly
        # Filter product_to_dressed so that it contains only state relevant to the two qubit computational states,
        # Also modify the original qubit index in the product indices to 0 and 1.
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
        if products_to_keep == None or products_to_keep == []:
            products_to_keep = list(
                product(*[range(dim) for dim in self.hilbertspace.subsystem_dims]))

        self.products_to_keep = products_to_keep
        self.diag_dressed_hamiltonian = self.truncate_function(qutip.Qobj((
            2 * np.pi * qutip.Qobj(np.diag(self.evals),
                                   dims=[self.hilbertspace.subsystem_dims] * 2)
        )[:, :]))

    def truncate_function(self, qobj):
        return truncate_custom(qobj, self.products_to_keep, self.product_to_dressed)

    def pad_back_function(self, qobj):
        return pad_back_custom(qobj, self.products_to_keep, self.product_to_dressed)

    def convert_dressed_to_product_vectorized(self,
                                             states,
                                             products_to_keep,
                                             num_processes=None,
                                             update_products_to_keep = True,
                                             show_progress=False):
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
                                    drive_terms: List[DriveTerm],
                                    c_ops: Union[None,
                                                 List[qutip.Qobj]] = None,
                                    e_ops: Union[None,
                                                 List[qutip.Qobj]] = None,

                                    post_processing=['pad_back'],
                                    ):
        '''
        This function runs mesolve on multiple initial states using multi-processing,
          and return a list of qutip.solver.result
        '''

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

        results = [None] * len(initial_states)
        with get_reusable_executor(max_workers=None, context='loky') as executor:
            futures = {executor.submit(ODEsolve_and_post_process,
                                       y0=initial_states[i],
                                       tlist=tlist,

                                       static_hamiltonian=self.diag_dressed_hamiltonian,
                                       drive_terms=drive_terms,
                                       c_ops=c_ops,
                                       e_ops=e_ops,
                                       post_processing_funcs=post_processing_funcs,
                                       post_processing_args=post_processing_args,
                                       ): i for i in range(len(initial_states))}

            for future in concurrent.futures.as_completed(futures):
                original_index = futures[future]
                results[original_index] = future.result()

        return results
    

    def run_dq_mesolve_parrallel(self,
                                 initial_states: qutip.Qobj,  # truncated initial states
                                 tlist: np.array,
                                 drive_terms: List[DriveTerm],
                                 c_ops: Union[None, List[qutip.Qobj]] = None,
                                 e_ops: Union[None, List[qutip.Qobj]] = None,

                                 post_processing=['pad_back'],
                                 ):
        #######################################################
        #     '''
        #     This function runs dq.mesolve or dq.sesolve using dq's parrellelism,
        #     then convert dq.Result into a list of qutip.Result
        #     and finally use cpu multiprocessing to do post processing steps
        #         like padding truncated hillbert space back to full dimension,
        #         or partial trace to get a qubit density matrix

        #     '''
        #     def _H(t):
        #         _H = jnp.array(self.diag_dressed_hamiltonian)
        #         for term in drive_terms:
        #             _H += jnp.array(term.driven_op)* term.pulse_shape_func(t, term.pulse_shape_args)
        #         return _H

        #     H =  timecallable(_H)

        #     if c_ops == [] or c_ops == None:
        #         result = dq.sesolve(
        #             H = H,
        #             psi0 = initial_states,
        #             tsave = tlist,
        #             exp_ops = e_ops,
        #             solver = dq.solver.Tsit5(
        #                     rtol= 1e-06,
        #                     atol= 1e-06,
        #                     safety_factor= 0.9,
        #                     min_factor= 0.2,
        #                     max_factor = 5.0,
        #                     max_steps = int(1e4*(tlist[-1]-tlist[0])),
        #                 )
        #             )
        #         print(result)
        #     else:
        #         result = dq.mesolve(
        #             H = H,
        #             jump_ops = c_ops,
        #             rho0 = initial_states,
        #             tsave = tlist,
        #             exp_ops = e_ops,
        #             solver = dq.solver.Tsit5(
        #                     rtol= 1e-06,
        #                     atol= 1e-06,
        #                     safety_factor= 0.9,
        #                     min_factor= 0.2,
        #                     max_factor = 5.0,
        #                     max_steps = int(1e4*(tlist[-1]-tlist[0])),
        #                 )
        #             )
        #         print(result)

        #     # Convert dq.Result to a list of qutip.solver.Result
        #     results = []
        #     for i in range(len(initial_states)):
        #         qt_result = qutip.solver.Result()
        #         qt_result.solver = 'dynamiqs'
        #         qt_result.times = tlist
        #         qt_result.expect = result.expects[i]
        #         qt_result.states = dq.to_qutip(result.states[i])
        #         qt_result.num_expect = len(e_ops) if isinstance(e_ops, list) else 0
        #         qt_result.num_collapse = len(c_ops) if isinstance(c_ops, list) else 0
        #         results.append(qt_result)

        #     post_processed_results = [None] * len(results)
        #     post_processing_funcs = []
        #     post_processing_args = []
        #     if 'pad_back' in post_processing:
        #         post_processing_funcs.append(pad_back_custom)
        #         post_processing_args.append((self.products_to_keep,
        #                             self.product_to_dressed))
        #     if 'partial_trace_computational_states' in post_processing:
        #         post_processing_funcs.append(dressed_to_2_level_dm)
        #         post_processing_args.append((
        #                                     self.product_to_dressed,
        #                                     self.qbt_position,
        #                                     self.filtered_product_to_dressed,
        #                                     self.sign_multiplier,
        #                                     None
        #                                     ))

        #     with get_reusable_executor(max_workers=None, context='loky') as executor:
        #         futures = {executor.submit(post_process,
        #                                     result = results[i],
        #                                     post_processing_funcs=post_processing_funcs,
        #                                     post_processing_args=post_processing_args,
        #                                     ): i for i in range(len(results))}

        #         for future in concurrent.futures.as_completed(futures):
        #             original_index = futures[future]
        #             post_processed_results[original_index] = future.result()

        #     return post_processed_results
        #######################################################
        pass


class FluxoniumOscillatorSystem(CoupledSystem):
    '''
    To model leakage detection of 12 fluxonium
    '''

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

                products_to_keep: List[List[int]] = None,
                ):
        '''
        Initialize objects before truncation
        '''
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

    def set_new_operators_after_setting_new_product_to_keep(self):
        self.a_trunc = self.truncate_function(self.a)
        # self.truncate_function(self.hilbertspace.op_in_dressed_eigenbasis(self.osc.n_operator))
        self.driven_operator = self.a_trunc + self.a_trunc.dag()
        self.set_new_kappa(self.kappa)

    def set_new_kappa(self, kappa):
        self.kappa = kappa
        self.c_ops = [np.sqrt(self.kappa) * self.a_trunc]

    def get_ladder_overlap_arr(self,resonator_creation_arr):
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
    


class FluxoniumTransmonSystem(CoupledSystem):
    '''
    To model leakage detection of 12 fluxonium
    '''

    def __init__(self,
                 fluxonium: scqubits.Fluxonium,
                 transmon: scqubits.Transmon,
                 computaional_states: str,  # = '0,1' or '1,2'
                 g_strength: float = 0.18,
                 products_to_keep: List[List[int]] = None,
                 ):
        '''
        Initialize objects before truncation
        '''

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
    '''
    To model leakage detection of 12 fluxonium
    '''

    def __init__(self,
                 fluxonium1: scqubits.Fluxonium,
                 fluxonium2: scqubits.Fluxonium,
                 transmon: scqubits.Transmon,
                 computaional_states: str,  # = '0,1' or '1,2'

                 g_f1f2: float = 0.1,
                 g_f1t: float = 0.1,
                 g_f2t:float=0.1,
                 products_to_keep: List[List[int]] = None,
                 ):

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
    
    def get_SATD_CZ_drive_terms(self,
                               ):
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

class TransmonOscillatorSystem(CoupledSystem):
    def __init__(self,
                 EJ: float = None,
                 EC: float = None,
                 ng: float = None,
                max_ql: int = 50,
                ncut: int = 50,
                qbt: scqubits.Transmon = None,

                Er: float = None,
                osc_level: float = 30,

                osc: scqubits.Oscillator = None,

                kappa=0.01,

                g_strength: float = None,

                products_to_keep: List[List[int]] = None,
                ):
        '''
        Initialize objects before truncation
        '''
        if qbt is not None:
            self.qbt = qbt
        else:
            self.qbt = scqubits.Transmon(EJ=EJ, EC=EC, ng=ng,max_ql=max_ql,ncut=ncut)
        
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
                         computaional_states=[0,1])

        self.a = qutip.Qobj(self.hilbertspace.op_in_dressed_eigenbasis(
            self.osc.annihilation_operator)[:, :])
        self.kappa = kappa
        self.set_new_operators_after_setting_new_product_to_keep()

    def set_new_operators_after_setting_new_product_to_keep(self):
        self.a_trunc = self.truncate_function(self.a)
        # self.truncate_function(self.hilbertspace.op_in_dressed_eigenbasis(self.osc.n_operator))
        self.driven_operator = self.a_trunc + self.a_trunc.dag()
        self.set_new_kappa(self.kappa)

    def set_new_kappa(self, kappa):
        self.kappa = kappa
        self.c_ops = [np.sqrt(self.kappa) * self.a_trunc]

    def get_ladder_overlap_arr(self,resonator_creation_arr):
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
    

# class FluxoniumOscillatorFilterSystem(CoupledSystem):
#     '''
#     To model leakage detection of 12 fluxonium with purcell filter
#     '''

#     def __init__(self,
#                  computaional_states: str,  # = '0,1' or '1,2'

#                  EJ: float = 2.33,
#                  EC: float = 0.69,
#                  EL: float = 0.12,
#                  qubit_level: float = 13,


#                  Er: float = 7.16518677,
#                  osc_level: float = 20,

#                  Ef: float = 7.13,
#                  filter_level: float = 7,
#                  # Ef *2pi = omega_f,  kappa_f = omega_f / Q , kappa_f^{-1} = 0.67 ns
#                  kappa_f=1.5,

#                  g_strength: float = 0.18,
#                  # G satisfies a relation with omega_r in equation 10 of Phys. Rev A 92. 012325 (2015)
#                  G_strength: float = 0.3,

#                  products_to_keep: List[List[int]] = None,
#                  w_d: float = None,
#                  ):

#         # Q_f = 30
#         # kappa_f = Ef * 2 * np.pi / Q_f
#         # kappa_r = 0.0001 #we want a really small effective readout resonator decay rate to reduce purcell decay
#         # G_strength =np.sqrt(kappa_f * kappa_r * ( 1 + (2*(Er-Ef)*2*np.pi/kappa_f )**2 ) /4)

#         self.G_strength = G_strength

#         self.qbt = scqubits.Fluxonium(
#             EJ=EJ, EC=EC, EL=EL, flux=0, cutoff=110, truncated_dim=qubit_level)
#         self.osc = scqubits.Oscillator(E_osc=Er, truncated_dim=osc_level)
#         self.filter = scqubits.Oscillator(E_osc=Ef, truncated_dim=filter_level)
#         hilbertspace = scqubits.HilbertSpace([self.qbt, self.osc, self.filter])
#         hilbertspace.add_interaction(
#             g_strength=g_strength, op1=self.qbt.n_operator, op2=self.osc.creation_operator, add_hc=True)
#         hilbertspace.add_interaction(g_strength=G_strength, op1=self.osc.creation_operator,
#                                      op2=self.filter.annihilation_operator, add_hc=True)

#         super().__init__(hilbertspace=hilbertspace,
#                          products_to_keep=products_to_keep,
#                          qbt_position=0,
#                          computaional_states=[int(computaional_states[0]), int(computaional_states[-1])])

#         self.a = qutip.Qobj(self.hilbertspace.op_in_dressed_eigenbasis(
#             self.osc.annihilation_operator)[:, :])
#         self.a_trunc = self.truncate_function(self.a)

#         self.b = qutip.Qobj(self.hilbertspace.op_in_dressed_eigenbasis(
#             self.filter.annihilation_operator)[:, :])
#         self.b_trunc = self.truncate_function(self.b)
#         self.driven_operator = self.b_trunc+self.b_trunc.dag()
#         self.c_ops = [np.sqrt(kappa_f) * self.b_trunc]

#         if w_d != None:
#             self.w_d = w_d
