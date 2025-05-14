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
from CoupledQuantumSystems.frame import RotatingFrame
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
    """
    driven_op: qutip.Qobj
    pulse_shape_func: Callable
    pulse_shape_args: Dict[str, float]

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
                return self.pulse_shape_func(t, unmodified_args,math=math)
            else:
                return self.pulse_shape_func(t, args,math=math)
        except KeyError as e:
            raise KeyError(f"Missing argument key for pulse_id {self.pulse_id}: {e}")
        except Exception as e:
            raise ValueError(f"Error processing pulse function for pulse_id {self.pulse_id}: {e}")

    def jax_wrapper(self) -> Callable:
        """Returns a function compatible with JAX (math=jnp)."""
        if not JAX_AVAILABLE:
            raise ImportError("JAX is not installed. Please install it using 'pip install CoupledQuantumSystems[jax]'")
        def jax_pulse_shape_func(t, args):
            return self.pulse_shape_func_with_id(t, args, math=jnp)
        return jax_pulse_shape_func

    def get_driven_op(self) -> qutip.Qobj:
        return self.driven_op

    def get_pulse_shape_func_with_id(self) -> Callable:
        return self.pulse_shape_func_with_id

    def get_pulse_shape_args_with_id(self) -> Dict[str, float]:
        return self.pulse_shape_args_with_id
    
    def get_pulse_shape_arg_val_without_id(self) -> Dict[str, float]:
        return self.pulse_shape_args_with_id
    
    def set_pulse_shape_arg_val_without_id(self,key,value):
        if self.pulse_id != None:
            self.pulse_shape_args_with_id[f"{key}{self.pulse_id}"] = value
        else:
            self.pulse_shape_args_with_id[f"{key}"] = value
    
    def visualize(self,ax,tlist,args,alpha=1,color=None,text = False):
        if color is None:
            color = 'blue'
        ax.plot(tlist, self.pulse_shape_func_with_id(tlist,args),label = self.pulse_id,alpha=alpha,color = color)
        if text:
            ax.text(tlist[int(len(tlist)/3)], 2*np.pi* 0.99* self.pulse_shape_args['amp'],f"{self.pulse_id} freq: {self.pulse_shape_args['w_d']}")

def rotating_wave_approximation(
        frame: RotatingFrame,
        drive_terms: List[DriveTerm],
        cutoff_freq: float = 1.0e9      # Hz
) -> List[DriveTerm]:
    """
    Filter each DriveTerm operator in *frame basis* and split into
    real / quadrature components (Hermitian).
    """
    ν_jk = frame.bohr_freqs
    out_terms: List[DriveTerm] = []

    def _is_zero(M: qutip.Qobj): return M.data.nnz == 0

    for term in drive_terms:
        ω_d = term.pulse_shape_args['w_d']          # Hz
        φ   = term.pulse_shape_args.get('phi', 0.0)
        Gf  = frame.to_frame_basis(term.driven_op).full()

        keep_pos = np.abs(+ω_d + ν_jk) < cutoff_freq
        keep_neg = np.abs(-ω_d + ν_jk) < cutoff_freq
        if not (keep_pos.any() or keep_neg.any()):
            continue

        G_pos = qutip.Qobj(Gf * keep_pos)
        G_neg = qutip.Qobj(Gf * keep_neg)
        G_c   = 0.5*(G_pos + G_neg)
        G_s   = 0.5j*(G_pos - G_neg)

        if not _is_zero(G_c):
            out_terms.append(
                DriveTerm(
                    driven_op       = frame.from_frame_basis(G_c).tidyup(1e-14),
                    pulse_shape_func= term.pulse_shape_func,
                    pulse_shape_args={**term.pulse_shape_args},
                    pulse_id        =(term.pulse_id or "")+"_rwa"
                )
            )
        if not _is_zero(G_s):
            out_terms.append(
                DriveTerm(
                    driven_op       = frame.from_frame_basis(G_s).tidyup(1e-14),
                    pulse_shape_func= term.pulse_shape_func,
                    pulse_shape_args={**term.pulse_shape_args,'phi':φ-np.pi/2},
                    pulse_id        =(term.pulse_id or "")+"_rwa_q"
                )
            )
    return out_terms

def square_pulse_with_rise_fall(t,
                                args = {}, math=np):
    
    w_d = args['w_d']
    amp = args['amp']
    t_start = args.get('t_start', 0)  # Default start time is 0
    t_rise = args.get('t_rise', 1e-13)  # Default rise time is 0 for no rise
    t_square = args.get('t_square', 0)  # Duration of constant amplitude
    phi = args.get('phi', 0)

    def cos_modulation():
        return 2 * math.pi * amp * math.cos(w_d * 2 * math.pi * t - phi)
    
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
    return (square_envelope + rise_envelope  + fall_envelope) * cos_modulation()

def sin_squared_pulse_with_modulation(t, args={},math=np):
    w_d = args['w_d']
    amp = args['amp']
    t_duration = args.get('t_duration')
    t_start = args.get('t_start', 0)  # Default start time is 0
    phi = args.get('phi', 0)

    def cos_modulation():
        return 2 * math.pi * math.cos(w_d * 2 * math.pi * t - phi)
    
    t_end = t_start + t_duration  # End of the pulse
    
    inside_window = (t >= t_start) & (t <= t_end)
    envelope = math.where(
        inside_window,
        math.sin(math.pi * (t - t_start) / t_duration) ** 2,
        0.0
    )
    return amp * envelope * cos_modulation()

def sin_squared_DRAG_with_modulation(t, args={},math=np):
    w_d = args['w_d']
    amp = args['amp']
    amp_correction = args['amp_correction'] # Usage: amp_ratio = -1/(2*np.pi* Delta), amp_correction = amp_ratio * amp
    t_duration = args.get('t_duration')
    t_start = args.get('t_start', 0)  # Default start time is 0
    phi = args.get('phi', 0)

    t_end = t_start + t_duration  # End of the pulse
    
    inside_window = (t >= t_start) & (t <= t_end)
    envelope = math.where(
        inside_window,
        math.sin(math.pi * (t - t_start) / t_duration) ** 2,
        0.0
    )
    envelope_derivative = math.where(
        inside_window,
        (math.pi/t_duration) *math.sin(2 * math.pi * (t - t_start) / t_duration),
        0.0
    )
    return 2 * math.pi *amp * envelope * math.cos(w_d * 2 * math.pi * t - phi) + \
        -1j * 2 * math.pi * amp_correction * envelope_derivative * math.cos(w_d * 2 * math.pi * t - phi)

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

def gaussian_pulse(t, args={},math=np):
    w_d = args['w_d']
    amp = args['amp']
    t_duration = args['t_duration']
    t_start = args.get('t_start', 0)  # Default start time is 0
    how_many_sigma = args.get('how_many_sigma', 6)  # Default factor to determine sigma
    normalize = args.get('normalize', False)  # Default normalization is False
    phi = args.get('phi', 0)
    sigma = t_duration/how_many_sigma
    t_center = t_start + t_duration / 2  # Center of the Gaussian pulse

    def gaussian(t):
        return amp * math.exp(-((t - t_center) ** 2) / (2 * sigma ** 2))
    def cos_modulation():
        return 2 * math.pi * math.cos(w_d * 2 * math.pi * t - phi)
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
    return envelope * cos_modulation() 

def gaussian_DRAG_pulse(t, args={},math=np):
    w_d = args['w_d']
    amp = args['amp']
    amp_correction_scaling_factor = args['amp_correction_scaling_factor']
    t_duration = args['t_duration']
    t_start = args.get('t_start', 0)  # Default start time is 0
    how_many_sigma = args.get('how_many_sigma', 6)  # Default factor to determine sigma
    normalize = args.get('normalize', False)  # Default normalization is False
    phi = args.get('phi', 0)
    sigma = t_duration/how_many_sigma
    t_center = t_start + t_duration / 2  # Center of the Gaussian pulse

    def gaussian(t):
        return amp * math.exp(-((t - t_center) ** 2) / (2 * sigma ** 2))
    def cos_modulation():
        return 2 * math.pi * math.cos(w_d * 2 * math.pi * t - phi)
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
    return (1+1j*amp_correction_scaling_factor*(-(t - t_center)/sigma**2))*envelope* cos_modulation() 

def STIRAP_with_modulation(t,args = {},math=np):
    # Symmetric Rydberg controlled-𝑍 gates with adiabatic pulses M. Saffman, I. I. Beterov, A. Dalal, E. J. Páez, and B. C. Sanders Phys. Rev. A 101, 062309 – Published 3 June 2020
    # Optimum pulse shapes for stimulated Raman adiabatic passage Phys. Rev. A 80, 013417 G. S. Vasilev, A. Kuhn, and N. V. Vitanov 2009
    w_d = args['w_d']
    amp = args['amp']
    t_stop = args['t_stop']
    stoke = args['stoke'] # Stoke is the first pulse, pump is the second
    t_start = args.get('t_start', 0)
    phi = args.get('phi', 0)

    def cos_modulation():
        return 2 * math.pi * amp * math.cos(w_d * 2 * math.pi * t - phi)
    
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
        return (hyper_Gaussian_F(t)- a)/(1-a)* math.cos(math.pi/2 * mono_increasing_f(t)) * cos_modulation()
    else:
        return (hyper_Gaussian_F(t)- a)/(1-a) * math.sin(math.pi/2 * mono_increasing_f(t)) * cos_modulation()
    

def Hyper_Gaussian_DRAG_STIRAP_stoke_jk(t, args, math=np):
    # stoke is the first pulse
    w_d   = args['w_d']
    amp   = args['amp']
    amp_correction = args['amp_correction']
    t_stop = args['t_stop']
    t_start = args.get('t_start', 0.0)
    phi   = args.get('phi', 0.0)
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
    return 2*math.pi* ( amp * env - 1j* amp_correction * d_env ) * math.cos(2*math.pi*w_d*t - phi)

def Hyper_Gaussian_DRAG_STIRAP_pump_ij(t, args, math=np):
    # stoke is the first pulse
    w_d   = args['w_d']
    amp   = args['amp']
    amp_correction = args['amp_correction']
    t_stop = args['t_stop']
    t_start = args.get('t_start', 0.0)
    phi   = args.get('phi', 0.0)
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
    return coeff * math.cos(2.0 * math.pi * w_d * t - phi)