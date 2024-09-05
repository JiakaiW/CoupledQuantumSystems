import concurrent
from itertools import product
from functools import partial
import multiprocessing

from loky import get_reusable_executor
import numpy as np
import qutip
import scqubits
from typing import List, Union, Tuple
from qobj_manip import *
from drive import *
from evo import ODEsolve_and_post_process
from qobj_manip import get_product, get_product_vectorized

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

        #############################################################################################
        #############################################################################################
        # TODO: This part about getting negative signs can be written more elegantly
        # Filter product_to_dressed so that it contains only state relevant to the two qubit computational states,
        # Also modify the original qubit index in the product indices to 0 and 1.
        self.filtered_product_to_dressed = {}
        for product_state, dressed_index in self.product_to_dressed.items():
            if product_state[qbt_position] in (self.computaional_states[0], self.computaional_states[1]):
                new_product_state = list(product_state)
                new_product_state[qbt_position] = 0 if product_state[qbt_position] == self.computaional_states[0] else 1
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
        #############################################################################################
        #############################################################################################

        self.set_new_product_to_keep(products_to_keep)

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
                                             num_processes=None):
        self.set_new_product_to_keep(products_to_keep)
        self.set_new_operators_after_setting_new_product_to_keep()
        
        if num_processes is None:
            num_processes = multiprocessing.cpu_count()
        
        # partial_function = partial(get_product,
        #                         pad_back_custom = self.pad_back_function,
        #                         product_to_dressed = self.product_to_dressed,
        #                         sign_multiplier = self.sign_multiplier)

        # with multiprocessing.Pool(processes=num_processes) as pool:
        #     product_states = pool.map(partial_function,states)

        # numpy already uses multi-core?
        product_states  = []
        for state in states:
            product_states.append(get_product_vectorized(state,
                                                         self.pad_back_custom,
                                                         self.product_to_dressed,
                                                         self.sign_multiplier))

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


class FluxoniumTunableTransmonSystem(CoupledSystem):
    '''
    To model leakage detection of 12 fluxonium
    '''

    def __init__(self,
                 fluxonium: scqubits.Fluxonium,
                 tune_tmon: scqubits.TunableTransmon,


                 computaional_states: str,  # = '0,1' or '1,2'

                 g_strength: float = 0.18,

                 products_to_keep: List[List[int]] = None,
                 w_d: float = None
                 ):
        '''
        Initialize objects before truncation
        '''

        self.fluxonium = fluxonium
        self.tune_tmon = tune_tmon
        hilbertspace = scqubits.HilbertSpace([self.fluxonium, self.tune_tmon])
        hilbertspace.add_interaction(
            g_strength=g_strength, op1=self.fluxonium.n_operator, op2=self.tune_tmon.n_operator, add_hc=False)

        super().__init__(hilbertspace=hilbertspace,
                         products_to_keep=products_to_keep,
                         qbt_position=0,
                         computaional_states=[int(computaional_states[0]), int(computaional_states[-1])])


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
                 w_d: float = None
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


class FluxoniumFluxoniumSystem(CoupledSystem):
    '''
    To model leakage detection of 12 fluxonium
    '''

    def __init__(self,
                 fluxonium1: scqubits.Fluxonium,
                 fluxonium2: scqubits.Fluxonium,
                 computaional_states: str,  # = '0,1' or '1,2'

                 g_strength: float = 0.18,

                 products_to_keep: List[List[int]] = None,
                 w_d: float = None
                 ):
        '''
        Initialize objects before truncation
        '''

        self.fluxonium1 = fluxonium1
        self.fluxonium2 = fluxonium2
        hilbertspace = scqubits.HilbertSpace(
            [self.fluxonium1, self.fluxonium2])
        hilbertspace.add_interaction(
            g_strength=g_strength, op1=self.fluxonium1.n_operator, op2=self.fluxonium2.n_operator, add_hc=False)

        super().__init__(hilbertspace=hilbertspace,
                         products_to_keep=products_to_keep,
                         qbt_position=0,
                         computaional_states=[int(computaional_states[0]), int(computaional_states[-1])])


class FluxoniumOscillatorSystem(CoupledSystem):
    '''
    To model leakage detection of 12 fluxonium
    '''

    def __init__(self,
                 computaional_states: str = '1,2',
                 drive_transition: Tuple[int] = None,

                 EJ: float = 2.33,
                 EC: float = 0.69,
                 EL: float = 0.12,
                 qubit_level: float = 13,


                 Er: float = 7.16518677,
                 osc_level: float = 30,
                 kappa=0.001,

                 g_strength: float = 0.18,

                 products_to_keep: List[List[int]] = None,
                 ):
        '''
        Initialize objects before truncation
        '''

        self.qbt = scqubits.Fluxonium(
            EJ=EJ, EC=EC, EL=EL, flux=0, cutoff=110, truncated_dim=qubit_level)
        # l_osc should have been 1/sqrt(2), otherwise I'm effectively reducing the coupling strength by sqrt(2)
        self.osc = scqubits.Oscillator(
            E_osc=Er, truncated_dim=osc_level, l_osc=1.0)
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


class FluxoniumOscillatorFilterSystem(CoupledSystem):
    '''
    To model leakage detection of 12 fluxonium with purcell filter
    '''

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
                 # Ef *2pi = omega_f,  kappa_f = omega_f / Q , kappa_f^{-1} = 0.67 ns
                 kappa_f=1.5,

                 g_strength: float = 0.18,
                 # G satisfies a relation with omega_r in equation 10 of Phys. Rev A 92. 012325 (2015)
                 G_strength: float = 0.3,

                 products_to_keep: List[List[int]] = None,
                 w_d: float = None,
                 ):

        # Q_f = 30
        # kappa_f = Ef * 2 * np.pi / Q_f
        # kappa_r = 0.0001 #we want a really small effective readout resonator decay rate to reduce purcell decay
        # G_strength =np.sqrt(kappa_f * kappa_r * ( 1 + (2*(Er-Ef)*2*np.pi/kappa_f )**2 ) /4)

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
