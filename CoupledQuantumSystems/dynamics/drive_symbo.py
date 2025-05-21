############################################################################
# drive.py – generic *only* DriveTerm
# ---------------------------------------------------------------------------
# A DriveTerm describes:  operator × [envelope × carrier].
# It is agnostic of *how* the envelope is produced.
# Users supply either
#   • pulse_type + symbolic_params (Preferred: uses create_pulse_shape for SymPy pure shapes)
#   • legacy_pulse_shape_func (numeric, full envelope)
# The class wraps these into:
#   – NumPy callable  for QuTiP / Dynamiqs
#   – JAX callable    for Qiskit‑Dynamics JIT path
#   – Qiskit Signal   (NumPy or JAX-based ScalableSymbolicPulse)
# No built‑in pulse shapes live here – they are in pulse_shapes_symbo.py.
############################################################################
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Any, List
import numpy as np
import qutip
import sympy as sp

# Import from local package
from .pulse_shapes_symbo import create_pulse_shape, PULSE_PARAM_SPECS, t_sym as default_t_sym

# optional JAX backend
try:
    import jax.numpy as jnp
    import jax
    JAX_AVAILABLE = True
except ImportError:
    jnp = None
    jax = None
    JAX_AVAILABLE = False

try:
    from qiskit_dynamics import Signal
    from qiskit import pulse
    from qiskit_dynamics.pulse import InstructionToSignals
    QISKIT_AVAILABLE = True
except ImportError:
    Signal = None # type: ignore
    pulse = None # type: ignore
    InstructionToSignals = None # type: ignore
    QISKIT_AVAILABLE = False

############################################################################
# Utility: LambdifiedExpression (REMOVED as direct lambdification is now simpler)
############################################################################

@dataclass
class DriveTermSymbolic:
    # --- Fields without defaults must come first ---
    driven_op: qutip.Qobj
    modulation_freq: float         # GHz, for the carrier
    amplitude: float               # Overall amplitude for the envelope
    duration: float                # Total duration of this pulse instance in seconds (for Qiskit)

    # --- Fields with defaults ---
    phi: float = 0.0               # rad, phase for the carrier e.g. cos(omega*t - phi)
    envelope_angle: float = 0.0    # Phase for the complex envelope, e.g., A * exp(i*envelope_angle) * shape(t)
    
    # --- Envelope Shape Definition (provide ONE of the two) ---
    # Option 1: Symbolic definition via pulse_type and shape-specific parameters
    pulse_type: Optional[str] = None                # E.g., "gaussian", "square_pulse_with_rise_fall"
    symbolic_params: Dict[sp.Symbol, float] = field(default_factory=dict) # Params for PURE shape, e.g. {pss.sigma_sym: 10e-9}

    # Option 2: User-provided callable for the full envelope (less common if using symbolic path)
    legacy_pulse_shape_func: Optional[Callable] = None # (t, args, math) -> complex float or float
    legacy_pulse_shape_args: Dict[str, float] = field(default_factory=dict)

    # --- Metadata & Qiskit specific (with defaults) ---
    pulse_id: Optional[str] = None
    qiskit_channel: Any = "d0" # Can be int or str like "d0"
    dt: Optional[float] = 0.222e-9 # Default dt for Qiskit sample conversion (seconds)

    # --- Internal fields (init=False) ---
    _internal_symbolic_expr_pure: Optional[sp.Expr] = field(init=False, default=None)
    _pure_shape_np_lambda: Optional[Callable] = field(init=False, default=None)
    _pure_shape_jx_lambda: Optional[Callable] = field(init=False, default=None)
    _ordered_shape_param_symbols: List[sp.Symbol] = field(init=False, default_factory=list)
    pulse_shape_func: Callable = field(init=False)


    # ------------------------------------------------------------------
    def __post_init__(self):
        if self.pulse_type is not None:
            if self.pulse_type not in PULSE_PARAM_SPECS:
                raise ValueError(f"Unknown pulse_type: {self.pulse_type}. Available: {list(PULSE_PARAM_SPECS.keys())}")

            pulse_meta_expr, pulse_meta_ordered_symbols, pulse_meta_t_sym = create_pulse_shape(
                pulse_type=self.pulse_type,
                params_values={s.name: v for s, v in self.symbolic_params.items()}
            )

            self._internal_symbolic_expr_pure = pulse_meta_expr
            self._ordered_shape_param_symbols = pulse_meta_ordered_symbols
            current_t_sym = pulse_meta_t_sym

            lambda_args = [current_t_sym] + self._ordered_shape_param_symbols
            
            self._pure_shape_np_lambda = sp.lambdify(lambda_args, self._internal_symbolic_expr_pure, modules=["numpy", "sympy"])
            
            if JAX_AVAILABLE:
                custom_jax_mappings = {
                    "exp": jnp.exp, 
                    "conjugate": jnp.conjugate, 
                    "Piecewise": jnp.select,
                    # Ensure sympy.nan (identified by its string name) is mapped to jax.numpy.nan
                    "nan": jnp.nan  
                }
                # Use custom mappings and the "jax" module. Removed "sympy" from this list
                # to prioritize JAX-native translations.
                self._pure_shape_jx_lambda = sp.lambdify(
                    lambda_args, 
                    self._internal_symbolic_expr_pure, 
                    modules=[custom_jax_mappings, "jax"]
                )
            else:
                self._pure_shape_jx_lambda = None

            def _symbolic_envelope_func(tt, args_ignored=None, *, math=np):
                numeric_shape_param_vals = [self.symbolic_params[sym] for sym in self._ordered_shape_param_symbols]
                
                pure_shape_val = 0.0
                if math is np:
                    if self._pure_shape_np_lambda is None:
                        raise RuntimeError("NumPy pure shape lambda not initialized.")
                    pure_shape_val = self._pure_shape_np_lambda(tt, *numeric_shape_param_vals)
                elif JAX_AVAILABLE and math is jnp:
                    if self._pure_shape_jx_lambda is None:
                        raise RuntimeError("JAX pure shape lambda not initialized.")
                    pure_shape_val = self._pure_shape_jx_lambda(tt, *numeric_shape_param_vals)
                else:
                    subs_dict = {current_t_sym: tt}
                    for i, sym_obj in enumerate(self._ordered_shape_param_symbols):
                        subs_dict[sym_obj] = numeric_shape_param_vals[i]
                    
                    if isinstance(tt, np.ndarray):
                        _evalf_scalar = lambda t_scalar: self._internal_symbolic_expr_pure.evalf(subs={**subs_dict, current_t_sym: t_scalar})
                        pure_shape_val = np.vectorize(_evalf_scalar)(tt)
                    else:
                        pure_shape_val = self._internal_symbolic_expr_pure.evalf(subs=subs_dict)

                complex_amp_factor = self.amplitude * math.exp(1j * self.envelope_angle)
                return complex_amp_factor * pure_shape_val 
            
            self.pulse_shape_func = _symbolic_envelope_func

        elif self.legacy_pulse_shape_func is not None:
            self.pulse_shape_func = self.legacy_pulse_shape_func
            self._internal_symbolic_expr_pure = None
            self._pure_shape_np_lambda = None
            self._pure_shape_jx_lambda = None
        else:
            raise ValueError("DriveTermSymbolic needs either pulse_type (for symbolic shapes) or legacy_pulse_shape_func.")

    # ------------------------------------------------------------
    # Coefficients for QuTiP / Dynamiqs (applies carrier)
    # ------------------------------------------------------------
    def _get_coeff_func(self, backend):
        current_args_for_legacy = self.legacy_pulse_shape_args if self.legacy_pulse_shape_func is not None else {}
        is_symbolic_path = self.pulse_type is not None

        def coeff(t, args_from_solver=None):
            if is_symbolic_path:
                envelope_val = self.pulse_shape_func(t, math=backend)
            else: # Legacy path
                envelope_val = self.pulse_shape_func(t, current_args_for_legacy, math=backend)
            
            mod_term = 2 * backend.pi * self.modulation_freq * t - self.phi
            return backend.real(envelope_val * backend.exp(1j * mod_term))

        return coeff

    def numpy_coeff(self) -> Callable:
        return self._get_coeff_func(np)

    def jax_coeff(self) -> Callable:
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX not installed for jax_coeff")
        return self._get_coeff_func(jnp)

    # ------------------------------------------------------------
    # Qiskit Signals
    # ------------------------------------------------------------
    def to_qiskit_signal_numpy(self) -> Signal: # type: ignore
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is not available. Please install qiskit and qiskit-dynamics.")
        is_symbolic_path = self.pulse_type is not None
        def qiskit_envelope_np(t_arr):
            if is_symbolic_path:
                return self.pulse_shape_func(t_arr, math=np)
            else: # Legacy path
                current_args_for_legacy = self.legacy_pulse_shape_args
                return self.pulse_shape_func(t_arr, current_args_for_legacy, math=np)

        return Signal(
            envelope=qiskit_envelope_np,
            carrier_freq=self.modulation_freq,
            phase=self.phi,
            name=self.pulse_id or f"{self.pulse_type}_drive_np" if self.pulse_type else "legacy_drive_np"
        )

    def to_qiskit_signal_jax(self) -> Signal: # type: ignore
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is not available. Please install qiskit and qiskit-dynamics.")
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX not installed for to_qiskit_signal_jax")

        if self.pulse_type is None or self._internal_symbolic_expr_pure is None or self._pure_shape_jx_lambda is None:
            print("Warning: Cannot create JAX signal directly. _pure_shape_jx_lambda not available or not a symbolic pulse. Falling back to numpy if possible.")
            # Fallback to numpy will likely fail if called from a JIT context with tracers.
            # For robust JAX usage, ensure pulse_type and symbolic setup are correct.
            return self.to_qiskit_signal_numpy()

        # _ordered_shape_param_symbols are the sympy symbols for shape params (e.g., pss.square_samples_sym)
        # self.symbolic_params maps these symbols to their values (which can be JAX tracers if duration is traced)
        # e.g., {pss.square_samples_sym: traced_duration_sec / self.dt}
        
        # These are the actual (potentially traced) values for the shape parameters
        # Ensure they are JAX arrays if they might be tracers from symbolic_params
        shape_param_traced_values = [
            jnp.asarray(self.symbolic_params[sym]) if isinstance(self.symbolic_params[sym], (jax.core.Tracer, jnp.ndarray)) 
            else self.symbolic_params[sym] 
            for sym in self._ordered_shape_param_symbols
        ]

        # self._pure_shape_jx_lambda expects its first argument as time in samples,
        # and subsequent arguments are the shape_param_traced_values.
        # The qiskit_dynamics.Signal envelope callable receives time in seconds.
        
        # self.dt should be a concrete float here for scaling.
        if not isinstance(self.dt, (float, int)) or self.dt <= 0:
             raise ValueError(f"self.dt must be a positive concrete float for JAX signal conversion, got {self.dt}, type {type(self.dt)}")

        # Ensure amplitude and envelope_angle are JAX-compatible if they could ever be traced.
        # For now, assuming they are concrete as per typical use.
        amp_val = jnp.asarray(self.amplitude, dtype=jnp.complex64) # qiskit-dynamics signals are complex
        env_angle_val = jnp.asarray(self.envelope_angle, dtype=jnp.float32)


        def jax_envelope_callable(t_seconds_array):
            # Convert time from seconds to samples
            # t_seconds_array can be a JAX array if generated from a traced duration.
            t_samples_array = t_seconds_array / self.dt
            
            # Call the JAX-lambdified pure shape function
            # _pure_shape_jx_lambda(t_samples, param1, param2, ...)
            pure_shape_values = self._pure_shape_jx_lambda(t_samples_array, *shape_param_traced_values)
            
            # Apply complex amplitude and phase
            # Ensure pure_shape_values is complex before multiplication if not already
            if not jnp.issubdtype(pure_shape_values.dtype, jnp.complexfloating):
                pure_shape_values = pure_shape_values.astype(jnp.complex64)

            complex_amp_factor = amp_val * jnp.exp(1j * env_angle_val)
            return complex_amp_factor * pure_shape_values

        return Signal(
            envelope=jax_envelope_callable,
            carrier_freq=self.modulation_freq, # This should be a concrete float
            phase=self.phi,                   # This should be a concrete float
            name=self.pulse_id or f"{self.pulse_type}_direct_jax_signal"
        )
