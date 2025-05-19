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
from qiskit_dynamics import Signal
from qiskit import pulse
from qiskit_dynamics.pulse import InstructionToSignals

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
                self._pure_shape_jx_lambda = sp.lambdify(lambda_args, self._internal_symbolic_expr_pure, modules=[{"exp": jnp.exp, "conjugate": jnp.conjugate, "Piecewise":jnp.select}, "jax", "sympy"])
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
    def to_qiskit_signal_numpy(self) -> Signal:
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

    def to_qiskit_signal_jax(self) -> Signal:
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX not installed for to_qiskit_signal_jax")
        if self.dt is None or self.dt <= 0:
            raise ValueError("dt must be a positive float for to_qiskit_signal_jax sample conversion.")
        
        if self.pulse_type is None or self._internal_symbolic_expr_pure is None:
            print("Warning: to_qiskit_signal_jax called without symbolic pulse_type. Falling back to numpy-based Signal.")
            return self.to_qiskit_signal_numpy()

        ssp_duration_samples = int(round(self.duration / self.dt))
        if ssp_duration_samples <= 0:
            raise ValueError(
                f"Calculated ScalableSymbolicPulse duration in samples ({ssp_duration_samples}) must be positive. "
                f"Duration in seconds: {self.duration}, dt: {self.dt}"
            )
        
        ssp_envelope_expr = self._internal_symbolic_expr_pure 
        ssp_parameters = {sym.name: float(val) for sym, val in self.symbolic_params.items()}
        # 1. Get the pure symbolic shape
        pure_expr = self._internal_symbolic_expr_pure 
        
        # 2. Handle Time Symbol and Scaling for the pure shape
        pulse_spec_entry = PULSE_PARAM_SPECS[self.pulse_type]

        time_sym_in_pure_expr = pulse_spec_entry.t_symbol # This is typically pss.t_sym (sp.Symbol('t'))
        
        # Ensure the time symbol is default_t_sym (sp.Symbol('t')) before scaling
        # default_t_sym is imported from pulse_shapes_symbo as t_sym
        if str(time_sym_in_pure_expr) != default_t_sym.name:
            pure_expr_using_default_t = pure_expr.subs({time_sym_in_pure_expr: default_t_sym})
        else:
            pure_expr_using_default_t = pure_expr

        # Scale time: t_samples -> t_samples * dt. default_t_sym is the symbol for t_samples here.
        scaled_time_pure_expr = pure_expr_using_default_t.subs({default_t_sym: default_t_sym * self.dt})

        # 3. Define Qiskit Amplitude/Angle Symbols
        q_amp_sym = sp.Symbol('amp')
        q_angle_sym = sp.Symbol('angle')

        # 4. Construct Full Qiskit Envelope, now including symbolic amp and angle
        # scaled_time_pure_expr is the f(t_samples*dt, shape_params_seconds)
        qiskit_ssp_envelope = q_amp_sym * sp.exp(sp.I * q_angle_sym) * scaled_time_pure_expr
        
        # 5. Populate Parameters Dictionary
        # Start with shape-specific parameters (values in seconds)
        ssp_parameters = {sym.name: float(val) for sym, val in self.symbolic_params.items()}
        # Add values for the new amp and angle symbols
        # ssp_parameters[q_amp_sym.name] = self.amplitude
        # ssp_parameters[q_angle_sym.name] = self.envelope_angle

        # 6. Instantiate ScalableSymbolicPulse
        this_pulse_instance = pulse.ScalableSymbolicPulse(
            pulse_type=self.pulse_id or f"{self.pulse_type}_ssp",
            duration=ssp_duration_samples,
            amp=self.amplitude,
            angle=self.envelope_angle, 
            parameters=ssp_parameters,
            envelope=qiskit_ssp_envelope,
            name=self.pulse_id or f"{self.pulse_type}_ssp_pulse"
        )
        
        channel_index = 0 
        if isinstance(self.qiskit_channel, str) and self.qiskit_channel.lower().startswith('d'):
            try:
                channel_index = int(self.qiskit_channel[1:])
            except ValueError:
                print(f"Warning: Could not parse channel index from qiskit_channel '{self.qiskit_channel}'. Defaulting to 0.")
        elif isinstance(self.qiskit_channel, int):
            channel_index = channel_index

        schedule_name = self.pulse_id or f"{self.pulse_type}_schedule"
        with pulse.build(name=schedule_name) as schedule_final:
            pulse.play(this_pulse_instance, pulse.DriveChannel(channel_index))
        
        q_channel_str_key = self.qiskit_channel if isinstance(self.qiskit_channel, str) else f"d{self.qiskit_channel}"
        converter = InstructionToSignals(
            dt=self.dt,
            carriers={q_channel_str_key: self.modulation_freq}
        )
        
        signals = converter.get_signals(schedule_final)
        if not signals:
            raise ValueError("No signals generated by InstructionToSignals.")
        
        final_signal = signals[0]
        if not isinstance(final_signal, Signal):
             raise TypeError(f"Generated signal is not a Qiskit Dynamics Signal. Type: {type(final_signal)}")

        final_signal.carrier_freq = self.modulation_freq
        final_signal.phase = self.phi
        
        return final_signal
