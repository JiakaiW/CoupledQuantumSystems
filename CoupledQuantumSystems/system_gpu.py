# import dynamiqs as dq
# import jax.numpy as jnp
# import qutip
# import numpy as np
# from typing import List, Union
# from CoupledQuantumSystems.drive import DriveTerm

# class GPUQuantumSystem:
#     def run_dq_mesolve_parrallel(self,
#                                  initial_states: qutip.Qobj,  # truncated initial states
#                                  tlist: np.array,
#                                  drive_terms: List[DriveTerm],
#                                  c_ops: Union[None, List[qutip.Qobj]] = None,
#                                  e_ops: Union[None, List[qutip.Qobj]] = None,
#                                 #  post_processing=['pad_back'],
#                                  ):
#         def _H(t):
#             _H = jnp.array(self.diag_hamiltonian)
#             for term in drive_terms:
#                 _H += jnp.array(term.driven_op)* term.pulse_shape_func(t, term.pulse_shape_args)
#             return _H

#         H =  dq.timecallable(_H)

#         if c_ops == [] or c_ops == None:
#             result = dq.sesolve(
#                 H = H,
#                 psi0 = initial_states,
#                 tsave = tlist,
#                 exp_ops = e_ops,
#                 solver = dq.solver.Tsit5(
#                         rtol= 1e-06,
#                         atol= 1e-06,
#                         safety_factor= 0.9,
#                         min_factor= 0.2,
#                         max_factor = 5.0,
#                         max_steps = int(1e4*(tlist[-1]-tlist[0])),
#                     )
#                 )
#             print(result)
#         else:
#             result = dq.mesolve(
#                 H = H,
#                 jump_ops = c_ops,
#                 rho0 = initial_states,
#                 tsave = tlist,
#                 exp_ops = e_ops,
#                 solver = dq.solver.Tsit5(
#                         rtol= 1e-06,
#                         atol= 1e-06,
#                         safety_factor= 0.9,
#                         min_factor= 0.2,
#                         max_factor = 5.0,
#                         max_steps = int(1e4*(tlist[-1]-tlist[0])),
#                     )
#                 )
#             print(result)

#         # Convert dq.Result to a list of qutip.solver.Result
#         results = []
#         for i in range(len(initial_states)):
#             qt_result = qutip.solver.Result()
#             qt_result.solver = 'dynamiqs'
#             qt_result.times = tlist
#             qt_result.expect = result.expects[i]
#             qt_result.states = dq.to_qutip(result.states[i])
#             qt_result.num_expect = len(e_ops) if isinstance(e_ops, list) else 0
#             qt_result.num_collapse = len(c_ops) if isinstance(c_ops, list) else 0
#             results.append(qt_result)
#         return results
#         # post_processed_results = [None] * len(results)
#         # post_processing_funcs = []
#         # post_processing_args = []
#         # if 'pad_back' in post_processing:
#         #     post_processing_funcs.append(pad_back_custom)
#         #     post_processing_args.append((self.products_to_keep,
#         #                         self.product_to_dressed))
#         # if 'partial_trace_computational_states' in post_processing:
#         #     post_processing_funcs.append(dressed_to_2_level_dm)
#         #     post_processing_args.append((
#         #                                 self.product_to_dressed,
#         #                                 self.qbt_position,
#         #                                 self.filtered_product_to_dressed,
#         #                                 self.sign_multiplier,
#         #                                 None
#         #                                 ))

#         # with get_reusable_executor(max_workers=None, context='loky') as executor:
#         #     futures = {executor.submit(post_process,
#         #                                 result = results[i],
#         #                                 post_processing_funcs=post_processing_funcs,
#         #                                 post_processing_args=post_processing_args,
#         #                                 ): i for i in range(len(results))}

#         #     for future in concurrent.futures.as_completed(futures):
#         #         original_index = futures[future]
#         #         post_processed_results[original_index] = future.result()

#         # return post_processed_results
