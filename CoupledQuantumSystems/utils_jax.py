
import jax.numpy as jnp
from flax import struct
import dynamiqs as dq
from dynamiqs import timecallable
# dq.set_precision( 'double')
import qcsys as qs
import jaxquantum as jqt
import jax
import math
from jax import jit, vmap

@struct.dataclass
class MyTransmon(qs.SingleChargeTransmon):
    '''
    The SingleChargeTransmon or Transmon in qcsys doesn't use the same hamiltonian as scqubit's
    I define this Transmon to keep it consistent with scqubit
    '''
    N_max_charge: int = struct.field(pytree_node=False)

    @classmethod
    def create(cls, N, N_max_charge, params, label=0, use_linear=False):
        return cls(N, N_max_charge, params, label, use_linear, N_max_charge)
    
    def get_H_full(self):
        #  consistant with scqubits 
        dimension = 2 * self.N_max_charge + 1
        def generate_hamiltonian_element(ind, Ec, N_max_charge, ng):
            return 4.0 * Ec * (ind - N_max_charge - ng) ** 2

        dim_range = jnp.arange(dimension)
        hamiltonian_mat = jnp.diag(vmap(generate_hamiltonian_element, in_axes=(0, None, None, None))(dim_range, self.params["Ec"], self.N_max_charge, self.params["ng"]))
        ind = jnp.arange(dimension - 1)
        hamiltonian_mat = hamiltonian_mat.at[ind, ind + 1].set(-self.params["Ej"] / 2.0)
        hamiltonian_mat = hamiltonian_mat.at[ind + 1, ind].set(-self.params["Ej"] / 2.0)
        hamiltonian_mat = jnp.array(hamiltonian_mat, dtype=jnp.complex128)
        H  = jqt.Qarray.create(hamiltonian_mat)
        # print(H.data)
        return  H
    def build_n_op(self):
        return jqt.Qarray.create(jnp.diag(jnp.arange(-self.N_max_charge, self.N_max_charge + 1)))
    @jit
    def get_op_in_H_eigenbasis(self, op):
        if type(op) == jqt.Qarray:
            op = op.data
        evecs = self.eig_systems["vecs"][:, : self.N]
        op = jnp.dot(jnp.conjugate(evecs.transpose()), jnp.dot(op, evecs))
        return jqt.Qarray.create(op)
    
@struct.dataclass
class MyResonator(qs.Device):
    @classmethod
    def create(cls, N, params, label=0, use_linear=False):
        return cls(N, N, params, label, use_linear)

    def common_ops(self):
        ops = {}

        N = self.N
        ops["id"] = jqt.identity(N)
        ops["a"] = jqt.destroy(N)
        ops["a_dag"] = jqt.create(N)
        ops["phi"] = (ops["a"] + ops["a_dag"]) / jnp.sqrt(2)
        ops["n"] = 1j * (ops["a_dag"] - ops["a"]) / jnp.sqrt(2)
        return ops

    def get_linear_ω(self):
        """Get frequency of linear terms."""
        return self.params["ω"]

    def get_H_linear(self):
        """Return linear terms in H."""
        w = self.get_linear_ω()
        return w * self.linear_ops["a_dag"] @ self.linear_ops["a"]

    def get_H_full(self):
        return self.get_H_linear()



############################################################################
#
#
# Functions about manipulating dynamiqs / jaxquantum / jax.numpy objects
#
#
############################################################################

# These are helper functions
def calculate_eig(Ns, H: jqt.Qarray):
    N_tot = math.prod(Ns)
    vals, kets = jnp.linalg.eigh(H.data)

    ketsT = kets.T

    def get_product_idx(edx):
        argmax = jnp.argmax(jnp.abs(ketsT[edx]))
        return  argmax  # product index
    edxs = jnp.arange(N_tot)
    product_indices_sorted_by_eval = vmap(get_product_idx)(edxs)
    return (vals,kets,product_indices_sorted_by_eval) # Here kets is equivalent to the S in qutip.Qobj.transform

def find_closest_dressed_index(product_index, product_indices_sorted_by_eval):
    dressed_index = jnp.argmin(jnp.abs(product_index - product_indices_sorted_by_eval))
    return dressed_index.item()

def transform_op_into_dressed_basis_jax(op_matrix: jqt.Qarray, 
                                        S: jax.Array) -> jax.Array:
    """
    Transform an operator into the dressed basis using JAX.

    Parameters:
    - op_matrix: A 2D JAX array representing the operator's matrix.
    - S: A 2D JAX array representing the dressed eigenvectors similar to the S in qutip.Qobj.transform

    Returns:
    - A 2D JAX array representing the transformed operator.
    """
    data = jnp.dot(S, jnp.dot(op_matrix.data, S.T.conj()))
    return data



############################################################################
#
#
# Ancilliary functions about pulse shaping and time dynamics
#
#
############################################################################

def square_pulse_with_rise_fall(t,
                                args = {}):
    
    w_d = args['w_d']
    amp = args['amp']
    t_start = args.get('t_start', 0)  # Default start time is 0
    t_rise = args.get('t_rise', 0)  # Default rise time is 0 for no rise
    t_square = args.get('t_square', 0)  # Duration of constant amplitude

    def cos_modulation():
        return 2 * jnp.pi * amp * jnp.cos(w_d * 2 * jnp.pi * t)
    
    t_fall_start = t_start + t_rise + t_square  # Start of fall
    t_end = t_fall_start + t_rise  # End of the pulse

    before_pulse_start = jnp.less(t, t_start)
    during_rise_segment = jnp.logical_and(jnp.greater(t_rise, 0), jnp.logical_and(jnp.greater_equal(t, t_start), jnp.less_equal(t, t_start + t_rise)))
    constant_amplitude_segment = jnp.logical_and(jnp.greater(t, t_start + t_rise), jnp.less_equal(t, t_fall_start))
    during_fall_segment = jnp.logical_and(jnp.greater(t_rise, 0), jnp.logical_and(jnp.greater(t, t_fall_start), jnp.less_equal(t, t_end)))

    return jnp.where(before_pulse_start, 0,
                    jnp.where(during_rise_segment, jnp.sin(jnp.pi * (t - t_start) / (2 * t_rise)) ** 2 * cos_modulation(),
                            jnp.where(constant_amplitude_segment, cos_modulation(),
                                        jnp.where(during_fall_segment, jnp.sin(jnp.pi * (t_end - t) / (2 * t_rise)) ** 2 * cos_modulation(), 0))))

def gaussian_pulse(t, args={}):
    # Area under envolope is amp * sigma * sqrt(2pi)
    # sigma = amp_with_2pi * pulse_length  /(   np.sqrt(2*np.pi)  *  amp_with_2pi )
    # t_tot = 8* sigma
    # sigma, t_tot
    w_d = args['w_d']
    amp = args['amp']
    t_start = args.get('t_start', 0)
    t_duration = args.get('t_duration', 0)
    sigma = args.get('sigma', 1)
    
    def cos_modulation(t_point):
        return 2 * jnp.pi * amp * jnp.cos(w_d * 2 * jnp.pi * t_point)
    
    t_end = t_start + t_duration

    def gaussian_envelope(t_point):
        return jnp.exp(-0.5 * ((t_point - (t_start + t_duration / 2)) ** 2) / sigma ** 2)
    
    pulse = jnp.where(jnp.logical_and(t >= t_start, t < t_end),
                     gaussian_envelope(t) * cos_modulation(t), 0)
    
    return pulse

def modified_drag_pulse(t, args):
    """
    Generate a modified DRAG pulse envelope using jax.numpy.

    Args:
        t (float): Time.
        args (dict): Dictionary containing pulse parameters.
            - 'duration': Pulse length.
            - 'sigma': Standard deviation of the Gaussian peak.
            - 'beta': Correction amplitude.
            - 'amp': Amplitude of the Gaussian envelope.

    Returns:
        complex: Modified DRAG pulse envelope at time t.
    """
    w_d = args['w_d']
    amp = args['amp']
    duration = args['duration']
    sigma = args['sigma']
    beta = args['beta']
    
    

    cos_modulation =  2 * jnp.pi * amp * jnp.cos(w_d * 2 * jnp.pi * t)
    
    a = jnp.exp(-0.5 * ((0 - duration / 2) / sigma) ** 2)
    gaussian = (jnp.exp(-0.5 * ((t - duration / 2) / sigma) ** 2) - a) / (1 - a)
    derivative = 1j * beta * (-1 / sigma ** 2) * (t - duration / 2) * gaussian
    modified_pulse =  (gaussian + derivative) *  cos_modulation
    return modified_pulse