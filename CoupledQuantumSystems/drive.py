############################################################################
#
#
# Ancilliary functions about pulse shaping and time dynamics
#
#
############################################################################
from dataclasses import dataclass, field
from typing import  Callable, Dict, List
import numpy as np
from qiskit_dynamics import Signal

try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
import qutip

class MathBackend:
    def __init__(self, backend):
        self.backend = backend

    def __getattr__(self, name):
        return getattr(self.backend, name)
        
@dataclass
class DriveTerm:
    """Wrapper for pulse shaping functions and their parameters.

    This class manages drive terms in quantum systems, providing a way to handle
    multiple pulses with the same shape function by using unique pulse IDs.

    Attributes:
        driven_op (qutip.Qobj): The operator being driven.
        pulse_shape_func (Callable): The function defining the pulse shape.
        pulse_shape_args (Dict[str, float]): Arguments for the pulse shape function.
        pulse_id (str, optional): Unique identifier for the pulse.
        modulation_freq (float): Frequency of the modulation (carrier frequency).
        phi (float): Phase of the modulation.
    """
    driven_op: qutip.Qobj
    pulse_shape_func: Callable
    pulse_shape_args: Dict[str, float]
    modulation_freq: float
    phi: float=0.0

    pulse_id: str = None
    pulse_shape_func_with_id: Callable = field(init=False)
    pulse_shape_args_with_id: Dict[str, float] = field(init=False)
    
    def __post_init__(self):
        if self.pulse_id != None:
            assert len(self.pulse_id)>0, 'cannot use pulse_id with zero length'
        self.pulse_shape_func_with_id = self.id_wrapper
        self.pulse_shape_args_with_id = self.modify_args_with_id(self.pulse_shape_args)

    def modify_args_with_id(self, pulse_shape_args: Dict[str, float]) -> Dict[str, float]:
        if self.pulse_id != None:
            return {f"{key}{self.pulse_id}": value for key, value in pulse_shape_args.items()}
        else:
            return {f"{key}": value for key, value in pulse_shape_args.items()}
    
    def id_wrapper(self, t, args={},math=np):
        try:
            if self.pulse_id is not None:
                # Remove the id from the args that contain id, and then call the original function
                unmodified_args = {key[:-len(self.pulse_id)]: value for key, value in args.items() if key.endswith(self.pulse_id)}
                envelope = self.pulse_shape_func(t, unmodified_args,math=math)
            else:
                envelope = self.pulse_shape_func(t, args,math=math)
            
            # Add modulation
            return envelope * math.cos(2 * math.pi * self.modulation_freq * t - self.phi)
        except KeyError as e:
            raise KeyError(f"Missing argument key for pulse_id {self.pulse_id}: {e}")
        except Exception as e:
            raise ValueError(f"Error processing pulse function for pulse_id {self.pulse_id}: {e}")

    def jax_wrapper(self) -> Callable:
        """Returns a function compatible with JAX (math=jnp)."""
        if not JAX_AVAILABLE:
            raise ImportError("JAX is not installed. Please install it using 'pip install CoupledQuantumSystems[jax]'")
        def jax_pulse_shape_func(t, args):
            return self.pulse_shape_func_with_id(t, args, math=jnp)*jnp.cos(2 * jnp.pi * self.modulation_freq * t - self.phi)
        return jax_pulse_shape_func

    def get_pulse_shape_args_with_id(self) -> Dict[str, float]:
        return self.pulse_shape_args_with_id
    
    def get_pulse_shape_arg_val_without_id(self) -> Dict[str, float]:
        return self.pulse_shape_args_with_id
    
    def visualize(self,ax,tlist,args,alpha=1,color=None,text = False):
        if color is None:
            color = 'blue'
        ax.plot(tlist, self.pulse_shape_func_with_id(tlist,args),label = self.pulse_id,alpha=alpha,color = color)
        if text:
            ax.text(tlist[int(len(tlist)/3)], 2*np.pi* 0.99* self.pulse_shape_args['amp'],f"{self.pulse_id} freq: {self.modulation_freq}")

    def envelope_to_qiskit_Signal(self)->Signal:
        """Convert this DriveTerm's envelope to a Qiskit Signal object.
        
        Returns:
            Signal: A Qiskit Signal object representing this drive term's envelope.
        """
        # Create a wrapper function that calls our pulse_shape_func_with_id
        def envelope_func(t):
            # Call our pulse shape function with the appropriate args
            return self.pulse_shape_func(t, self.pulse_shape_args)
        
        # Create and return the Signal object
        return Signal(
            envelope=envelope_func,
            carrier_freq=self.modulation_freq,
            phase=self.phi,
            name=self.pulse_id
        )

def square_pulse_with_rise_fall_envelope(t,
                                args = {}, math=np):
    """Envelope function for a square pulse with rise and fall times.
    
    Args:
        t: Time points
        args: Dictionary containing:
            amp: Amplitude of the pulse
            t_start: Start time of the pulse (default: 0)
            t_rise: Rise time (default: 1e-13)
            t_square: Duration of constant amplitude
        math: Math backend (numpy or jax.numpy)
    
    Returns:
        The envelope function value at time t
    """
    amp = args['amp']
    t_start = args.get('t_start', 0)  # Default start time is 0
    t_rise = args.get('t_rise', 1e-13)  # Default rise time is 0 for no rise
    t_square = args.get('t_square', 0)  # Duration of constant amplitude
    
    t_fall_start = t_start + t_rise + t_square  # Start of fall
    t_end = t_fall_start + t_rise  # End of the pulse
    
    rise_window = (t >= t_start) & (t <= t_start + t_rise)
    square_window = (t >= t_start + t_rise) & (t <= t_fall_start)
    fall_window = (t >= t_fall_start) & (t <= t_end)

    square_envelope = math.where(
        square_window,
        math.ones_like(t),
        0.0
    )
    rise_envelope = math.where(
        rise_window,
        math.sin(math.pi * (t - t_start) / (2 * t_rise)) ** 2,
        0.0
    )
    fall_envelope = math.where(
        fall_window,
        math.sin(math.pi * (t_end - t) / (2 * t_rise)) ** 2,
        0.0
    )
    return 2*math.pi*amp * (square_envelope + rise_envelope + fall_envelope)

def sin_squared_pulse_envelope(t, args={},math=np):
    """Envelope function for a sin-squared pulse.
    
    Args:
        t: Time points
        args: Dictionary containing:
            amp: Amplitude of the pulse
            t_duration: Duration of the pulse
            t_start: Start time of the pulse (default: 0)
        math: Math backend (numpy or jax.numpy)
    
    Returns:
        The envelope function value at time t
    """
    amp = args['amp']
    t_duration = args['t_duration']
    t_start = args.get('t_start', 0)  # Default start time is 0
    
    t_end = t_start + t_duration  # End of the pulse
    
    inside_window = (t >= t_start) & (t <= t_end)
    envelope = math.where(
        inside_window,
        math.sin(math.pi * (t - t_start) / t_duration) ** 2,
        0.0
    )
    return 2 * math.pi * amp * envelope

def sin_squared_DRAG_envelope(t, args={},math=np):
    """Envelope function for a sin-squared DRAG pulse.
    
    Args:
        t: Time points
        args: Dictionary containing:
            amp: Amplitude of the pulse
            amp_correction: Amplitude correction for DRAG
            t_duration: Duration of the pulse
            t_start: Start time of the pulse (default: 0)
        math: Math backend (numpy or jax.numpy)
    
    Returns:
        The complex envelope function value at time t
    """
    amp = args['amp']
    amp_correction = args['amp_correction'] # Usage: amp_ratio = -1/(2*np.pi* Delta), amp_correction = amp_ratio * amp
    t_duration = args['t_duration']
    t_start = args.get('t_start', 0)  # Default start time is 0
    
    t_end = t_start + t_duration  # End of the pulse
    
    inside_window = (t >= t_start) & (t <= t_end)
    envelope = math.where(
        inside_window,
        math.sin(math.pi * (t - t_start) / t_duration) ** 2,
        0.0
    )
    envelope_derivative = math.where(
        inside_window,
        (math.pi/t_duration) * math.sin(2 * math.pi * (t - t_start) / t_duration),
        0.0
    )
    return 2 * math.pi * (amp * envelope - 1j * amp_correction * envelope_derivative)

def recursive_drag_coeffs(delta1, delta2, amp):
    """
    Parameters
    ----------
    delta1, delta2 : float
        Detunings |ω_unwanted - ω_drive|  **in rad/s**  (use 2π·MHz).
    amp            : float
        Base in-phase amplitude used for the main envelope.

    Returns
    -------
    kappa1, kappa2 : floats
        Coefficients that multiply the first and second envelope
        derivatives, respectively, following Li et al. (2024).

    Formula (Li, Appendix D):
        κ₂ = - A / (Δ₁ Δ₂)
        κ₁ =   A (Δ₁ + Δ₂) / (Δ₁ Δ₂)
    """
    kappa2 = -amp / (delta1 * delta2)
    kappa1 =  amp * (delta1 + delta2) / (delta1 * delta2)
    return kappa1, kappa2


def sin_squared_recursive_DRAG(t, args=None, math=np):
    """
    Recursive‑DRAG pulse that punches out *two* notches.

    Required keys in `args`
    -----------------------
    w_d      : drive **frequency** in Hz (not radians/s)
    amp      : real in‑phase amplitude A
    delta1   : detuning Δ₁  (rad/s) of 1st unwanted transition
    delta2   : detuning Δ₂  (rad/s) of 2nd unwanted transition
               (use ±2π·MHz etc.; sign does not matter)
    t_duration : pulse length (s)

    Optional keys
    -------------
    t_start : left edge of pulse window (default 0 s)
    phi     : global phase of the carrier (default 0)
    """

    # fall‑back so args can be omitted in interactive calls
    if args is None:
        args = {}

    w_d        = args['w_d']
    amp        = args['amp']
    delta1     = args['delta1']
    delta2     = args['delta2']
    t_duration = args['t_duration']

    t_start = args.get('t_start', 0.0)
    phi     = args.get('phi', 0.0)

    # ---- DRAG coefficients that set the two spectral zeros ----
    kappa1, kappa2 = recursive_drag_coeffs(delta1, delta2, amp)

    # ---- handy aliases ----
    two_pi_over_T  = 2.0 * math.pi / t_duration
    pi_over_T      = math.pi / t_duration

    # ---- rectangular window ----
    inside = (t >= t_start) & (t <= t_start + t_duration)

    # ---- sin² envelope and its first two derivatives ----
    envelope = math.where(
        inside,
        math.sin(math.pi * (t - t_start) / t_duration) ** 2,
        0.0
    )

    # 1st derivative:  (π/T)·sin(2πτ)  with τ = (t‑t₀)/T
    envelope_d1 = math.where(
        inside,
        pi_over_T * math.sin(two_pi_over_T * (t - t_start)),
        0.0
    )

    # 2nd derivative:  (2π/T)² · (cos(2πτ) – 1)
    #  (the "–1" keeps the derivative continuous at the edges)
    envelope_d2 = math.where(
        inside,
        2.0 * (pi_over_T ** 2) *
        (math.cos(two_pi_over_T * (t - t_start)) - 1.0),
        0.0
    )

    # ---- complex base‑band envelope ----
    baseband = (
        amp * envelope +
        1j * kappa1 * envelope_d1 +
        kappa2 * envelope_d2
    )

    # ---- up‑convert with cosine modulation (∝ Re{…}) ----
    modulation = math.cos(2.0 * math.pi * w_d * t - phi)

    return baseband * modulation

def gaussian_pulse_envelope(t, args={},math=np):
    """Envelope function for a Gaussian pulse.
    
    Args:
        t: Time points
        args: Dictionary containing:
            amp: Amplitude of the pulse
            t_duration: Duration of the pulse
            t_start: Start time of the pulse (default: 0)
            how_many_sigma: Number of standard deviations to include (default: 6)
            normalize: Whether to normalize the pulse (default: False)
        math: Math backend (numpy or jax.numpy)
    
    Returns:
        The envelope function value at time t
    """
    amp = args['amp']
    t_duration = args['t_duration']
    t_start = args.get('t_start', 0)  # Default start time is 0
    how_many_sigma = args.get('how_many_sigma', 6)  # Default factor to determine sigma
    normalize = args.get('normalize', False)  # Default normalization is False
    
    sigma = t_duration/how_many_sigma
    t_center = t_start + t_duration / 2  # Center of the Gaussian pulse

    def gaussian(t):
        return amp * math.exp(-((t - t_center) ** 2) / (2 * sigma ** 2))
    
    t_end = t_start + t_duration  # End of the pulse

    inside_window = (t >= t_start) & (t <= t_end)
    if normalize:
        envelope = math.where(
            inside_window,
            gaussian(t),
            0.0
        )
        a = gaussian(t_start)
        envelope = (envelope - a) / (1 - a)
    else:
        envelope = math.where(
            inside_window,
            gaussian(t),
            0.0
        )
    return 2 * math.pi * envelope

def gaussian_DRAG_pulse_envelope(t, args={},math=np):
    """Envelope function for a Gaussian DRAG pulse.
    
    Args:
        t: Time points
        args: Dictionary containing:
            amp: Amplitude of the pulse
            amp_correction_scaling_factor: Scaling factor for DRAG correction
            t_duration: Duration of the pulse
            t_start: Start time of the pulse (default: 0)
            how_many_sigma: Number of standard deviations to include (default: 6)
            normalize: Whether to normalize the pulse (default: False)
        math: Math backend (numpy or jax.numpy)
    
    Returns:
        The complex envelope function value at time t
    """
    amp = args['amp']
    amp_correction_scaling_factor = args['amp_correction_scaling_factor']
    t_duration = args['t_duration']
    t_start = args.get('t_start', 0)  # Default start time is 0
    how_many_sigma = args.get('how_many_sigma', 6)  # Default factor to determine sigma
    normalize = args.get('normalize', False)  # Default normalization is False
    
    sigma = t_duration/how_many_sigma
    t_center = t_start + t_duration / 2  # Center of the Gaussian pulse

    def gaussian(t):
        return amp * math.exp(-((t - t_center) ** 2) / (2 * sigma ** 2))
    
    t_end = t_start + t_duration  # End of the pulse

    inside_window = (t >= t_start) & (t <= t_end)
    envelope = math.where(
        inside_window,
        gaussian(t),
        0.0
    )
    if normalize:
        a = gaussian(t_start)
        envelope = (envelope - a) / (1 - a)
    
    # Add DRAG correction
    drag_correction = 1 + 1j * amp_correction_scaling_factor * (-(t - t_center)/sigma**2)
    return 2 * math.pi * drag_correction * envelope

def STIRAP_envelope(t, args={}, math=np):
    """Envelope function for a STIRAP pulse.
    # Symmetric Rydberg controlled-𝑍 gates with adiabatic pulses M. Saffman, I. I. Beterov, A. Dalal, E. J. Páez, and B. C. Sanders Phys. Rev. A 101, 062309 – Published 3 June 
        2020
    # Optimum pulse shapes for stimulated Raman adiabatic passage Phys. Rev. A 80, 013417 G. S. Vasilev, A. Kuhn, and N. V. Vitanov 2009

    Args:
        t: Time points
        args: Dictionary containing:
            amp: Amplitude of the pulse
            t_stop: Stop time of the pulse
            stoke: Whether this is a stoke pulse (True) or pump pulse (False)
            t_start: Start time of the pulse (default: 0)
        math: Math backend (numpy or jax.numpy)
    
    Returns:
        The envelope function value at time t
    """
    amp = args['amp']
    t_stop = args['t_stop']
    stoke = args['stoke'] # Stoke is the first pulse, pump is the second
    t_start = args.get('t_start', 0)

    
    lambda_val = 4
    tau_for_mono = (t_stop-t_start) / 6
    center = (t_stop-t_start) / 2 + t_start

    def mono_increasing_f(t):
        return 1 / (1 + math.exp(-lambda_val * (t-center) / tau_for_mono))
    
    def hyper_Gaussian_F(t):
        T0 = 2 * tau_for_mono
        return math.exp(  - ((t-center) / T0) ** (2*3) )
    a = hyper_Gaussian_F(t_start)
    if stoke:
        return (hyper_Gaussian_F(t)- a)/(1-a)* math.cos(math.pi/2 * mono_increasing_f(t)) * 2 * math.pi * amp
    else:
        return (hyper_Gaussian_F(t)- a)/(1-a) * math.sin(math.pi/2 * mono_increasing_f(t)) * 2 * math.pi * amp
    

def Hyper_Gaussian_DRAG_STIRAP_stoke_jk_envelope(t, args, math=np):
    # stoke is the first pulse
    amp   = args['amp']
    amp_correction = args['amp_correction']
    t_stop = args['t_stop']
    t_start = args.get('t_start', 0.0)
    # --- helpers reused from your code -----------------------------
    λ         = 4.0
    τ_mono    = (t_stop - t_start) / 6.0
    centre    = (t_stop + t_start) / 2.0 + t_start
    def mono(t):
        return 1.0 / (1.0 + math.exp(-λ * (t - centre) / τ_mono))
    def hyperG(t):    # hyper‑Gaussian
        T0 = 2*τ_mono
        return math.exp(-((t-centre)/T0)**6)
    def d_hyperG(t):
        T0 = 2*τ_mono
        return hyperG(t) * (-6) * ((t - centre)/T0)**5 / T0
    a0 = hyperG(t_start)
    def envelope(t):
        return (hyperG(t) - a0) / (1 - a0) * math.cos(math.pi/2 * mono(t))
    def d_envelope(t):
        term1 = d_hyperG(t) / (1 - a0)
        term2 = -(hyperG(t) - a0) / (1 - a0)**2 * d_hyperG(t_start)
        # product rule with the mono() factor
        d_mono = (λ / τ_mono) * mono(t) * (1 - mono(t))
        return (term1 + term2) * np.cos(np.pi/2 * mono(t)) \
            + (hyperG(t) - a0)/(1 - a0) * (-np.pi/2) * np.sin(np.pi/2*mono(t)) * d_mono
    env = envelope(t)
    d_env = d_envelope(t)
    return 2*math.pi* ( amp * env - 1j* amp_correction * d_env )

def Hyper_Gaussian_DRAG_STIRAP_pump_ij_envelope(t, args, math=np):
    # stoke is the first pulse
    amp   = args['amp']
    amp_correction = args['amp_correction']
    t_stop = args['t_stop']
    t_start = args.get('t_start', 0.0)
    dt = 1e-12
    # --- helpers reused from your code -----------------------------
    λ         = 4.0
    τ_mono    = (t_stop - t_start) / 6.0
    centre    = (t_stop + t_start) / 2.0 + t_start
    def mono(t):
        return 1.0 / (1.0 + math.exp(-λ * (t - centre) / τ_mono))
    def hyperG(t):
        T0 = 2.0 * τ_mono
        return math.exp(-((t - centre) / T0) ** 6)
    def d_hyperG(t):
        T0 = 2.0 * τ_mono
        return hyperG(t) * (-6.0) * ((t - centre) / T0) ** 5 / T0
    a0 = hyperG(t_start)
    # pump uses  sin(π mono/2)
    def envelope(t):
        return (hyperG(t) - a0) / (1.0 - a0) * math.sin(math.pi / 2.0 * mono(t))
    def d_envelope(t):
        term1 = d_hyperG(t) / (1.0 - a0)
        term2 = -(hyperG(t) - a0) / (1.0 - a0) ** 2 * d_hyperG(t_start)
        d_mono = (λ / τ_mono) * mono(t) * (1.0 - mono(t))
        return (
            (term1 + term2) * math.sin(math.pi / 2.0 * mono(t))
            + (hyperG(t) - a0)
              / (1.0 - a0)
              * (math.pi / 2.0)
              * math.cos(math.pi / 2.0 * mono(t))
              * d_mono
        )
    env   = envelope(t)
    d_env = d_envelope(t)
    # ----------- full complex coefficient ----------------------------------------
    coeff = 2.0 * math.pi * (amp * env - 1j * amp_correction * d_env)
    # return coefficient multiplied by carrier (σₓ term)
    return coeff 