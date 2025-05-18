# This was used to construct objects for the dynamiqs backend. Now I use the objects in systems.py to construct the objects


# import jax.numpy as jnp
# from flax import struct
# import dynamiqs as dq
# from dynamiqs import timecallable
# # dq.set_precision( 'double')
# import qcsys as qs
# import jaxquantum as jqt
# import jax
# import math
# from jax import jit, vmap

# @struct.dataclass
# class MyTransmon(qs.SingleChargeTransmon):
#     '''
#     The SingleChargeTransmon or Transmon in qcsys doesn't use the same hamiltonian as scqubit's
#     I define this Transmon to keep it consistent with scqubit
#     '''
#     N_max_charge: int = struct.field(pytree_node=False)

#     @classmethod
#     def create(cls, N, N_max_charge, params, label=0, use_linear=False):
#         return cls(N, N_max_charge, params, label, use_linear, N_max_charge)
    
#     def get_H_full(self):
#         #  consistant with scqubits 
#         dimension = 2 * self.N_max_charge + 1
#         def generate_hamiltonian_element(ind, Ec, N_max_charge, ng):
#             return 4.0 * Ec * (ind - N_max_charge - ng) ** 2

#         dim_range = jnp.arange(dimension)
#         hamiltonian_mat = jnp.diag(vmap(generate_hamiltonian_element, in_axes=(0, None, None, None))(dim_range, self.params["Ec"], self.N_max_charge, self.params["ng"]))
#         ind = jnp.arange(dimension - 1)
#         hamiltonian_mat = hamiltonian_mat.at[ind, ind + 1].set(-self.params["Ej"] / 2.0)
#         hamiltonian_mat = hamiltonian_mat.at[ind + 1, ind].set(-self.params["Ej"] / 2.0)
#         hamiltonian_mat = jnp.array(hamiltonian_mat, dtype=jnp.complex128)
#         H  = jqt.Qarray.create(hamiltonian_mat)
#         # print(H.data)
#         return  H
#     def build_n_op(self):
#         return jqt.Qarray.create(jnp.diag(jnp.arange(-self.N_max_charge, self.N_max_charge + 1)))
#     @jit
#     def get_op_in_H_eigenbasis(self, op):
#         if type(op) == jqt.Qarray:
#             op = op.data
#         evecs = self.eig_systems["vecs"][:, : self.N]
#         op = jnp.dot(jnp.conjugate(evecs.transpose()), jnp.dot(op, evecs))
#         return jqt.Qarray.create(op)
    
# @struct.dataclass
# class MyResonator(qs.Device):
#     @classmethod
#     def create(cls, N, params, label=0, use_linear=False):
#         return cls(N, N, params, label, use_linear)

#     def common_ops(self):
#         ops = {}

#         N = self.N
#         ops["id"] = jqt.identity(N)
#         ops["a"] = jqt.destroy(N)
#         ops["a_dag"] = jqt.create(N)
#         ops["phi"] = (ops["a"] + ops["a_dag"]) / jnp.sqrt(2)
#         ops["n"] = 1j * (ops["a_dag"] - ops["a"]) / jnp.sqrt(2)
#         return ops

#     def get_linear_ω(self):
#         """Get frequency of linear terms."""
#         return self.params["ω"]

#     def get_H_linear(self):
#         """Return linear terms in H."""
#         w = self.get_linear_ω()
#         return w * self.linear_ops["a_dag"] @ self.linear_ops["a"]

#     def get_H_full(self):
#         return self.get_H_linear()



# ############################################################################
# #
# #
# # Functions about manipulating dynamiqs / jaxquantum / jax.numpy objects
# #
# #
# ############################################################################

# # These are helper functions
# def calculate_eig(Ns, H: jqt.Qarray):
#     N_tot = math.prod(Ns)
#     vals, kets = jnp.linalg.eigh(H.data)

#     ketsT = kets.T

#     def get_product_idx(edx):
#         argmax = jnp.argmax(jnp.abs(ketsT[edx]))
#         return  argmax  # product index
#     edxs = jnp.arange(N_tot)
#     product_indices_sorted_by_eval = vmap(get_product_idx)(edxs)
#     return (vals,kets,product_indices_sorted_by_eval) # Here kets is equivalent to the S in qutip.Qobj.transform

# def find_closest_dressed_index(product_index, product_indices_sorted_by_eval):
#     dressed_index = jnp.argmin(jnp.abs(product_index - product_indices_sorted_by_eval))
#     return dressed_index.item()

# def transform_op_into_dressed_basis_jax(op_matrix: jqt.Qarray, 
#                                         S: jax.Array) -> jax.Array:
#     """
#     Transform an operator into the dressed basis using JAX.

#     Parameters:
#     - op_matrix: A 2D JAX array representing the operator's matrix.
#     - S: A 2D JAX array representing the dressed eigenvectors similar to the S in qutip.Qobj.transform

#     Returns:
#     - A 2D JAX array representing the transformed operator.
#     """
#     data = jnp.dot(S, jnp.dot(op_matrix.data, S.T.conj()))
#     return data

