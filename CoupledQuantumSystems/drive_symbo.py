############################################################################
# drive.py – generic *only* DriveTerm
# ---------------------------------------------------------------------------
# A DriveTerm describes:  operator × [envelope × carrier].
# It is agnostic of *how* the envelope is produced.
# Users supply either
#   • pulse_shape_func  (numeric)
#   • symbolic_expr + params (SymPy)
# The class merely wraps these into:
#   – NumPy callable  for QuTiP / Dynamiqs
#   – JAX callable    for Qiskit‑Dynamics JIT path
#   – Qiskit Signal   (NumPy or JAX)
# No built‑in pulse shapes live here – keep them in pulse_shapes.py or elsewhere.
############################################################################
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Any
import numpy as np
import qutip
import sympy as sp
from qiskit_dynamics import Signal
from qiskit import pulse
from qiskit_dynamics.pulse import InstructionToSignals
import functools

# optional JAX backend
try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jnp = None
    JAX_AVAILABLE = False

############################################################################
# Utility: build lambdas from symbolic expression
############################################################################

class LambdifiedExpression:
    """Descriptor to lambdify symbolic expression with cache.
    
    When a new symbolic expression is assigned for the first time, this class
    will internally lambdify the expressions and store the resulting callbacks in its cache.
    The next time it encounters the same expression it will return the cached callbacks
    thereby increasing the code's speed.
    """
    def __init__(self, attribute: str):
        """Create new descriptor.
        
        Args:
            attribute: Name of attribute that returns the target expression to evaluate.
        """
        self.attribute = attribute
        self.lambda_funcs: dict[int, Callable] = {}

    def __get__(self, instance, owner) -> Callable:
        expr = getattr(instance, self.attribute, None)
        if expr is None:
            raise ValueError(f"'{self.attribute}' is not assigned.")
        key = hash(expr)
        if key not in self.lambda_funcs:
            self.__set__(instance, expr)
        return self.lambda_funcs[key]

    def __set__(self, instance, value):
        key = hash(value)
        if key not in self.lambda_funcs:
            params = []
            for p in sorted(value.free_symbols, key=lambda s: s.name):
                if p.name == "t":
                    # Argument "t" must be placed at first. This is a vector.
                    params.insert(0, p)
                    continue
                params.append(p)

            try:
                # Use sympy's lambdify for better compatibility
                lamb = sp.lambdify(params, [value], modules=["numpy"])
                
                def _wrapped_lamb(*args):
                    if isinstance(args[0], np.ndarray):
                        # When the args[0] is a vector ("t"), tile other arguments args[1:]
                        # to prevent evaluation from looping over each element in t.
                        t = args[0]
                        args = np.hstack(
                            (
                                t.reshape(t.size, 1),
                                np.tile(args[1:], t.size).reshape(t.size, len(args) - 1),
                            )
                        )
                    return lamb(args)

                self.lambda_funcs[key] = _wrapped_lamb
            except Exception as e:
                raise ValueError(f"Failed to lambdify expression: {e}")

def _get_expression_args(expr: sp.Expr, params: dict[str, float]) -> list[np.ndarray | float]:
    """Get arguments to evaluate expression.
    
    Args:
        expr: Symbolic expression to evaluate.
        params: Dictionary of parameter values.
        
    Returns:
        Arguments passed to the lambdified expression.
        
    Raises:
        ValueError: When a free symbol value is not defined in the parameters.
    """
    args = []
    for symbol in sorted(expr.free_symbols, key=lambda s: s.name):
        if symbol.name == "t":
            # 't' is a special parameter to represent time vector.
            # This should be placed at first to broadcast other parameters.
            times = np.arange(0, params["duration"]) + 1 / 2
            args.insert(0, times)
            continue
        try:
            args.append(params[symbol.name])
        except KeyError as ex:
            raise ValueError(
                f"Pulse parameter '{symbol.name}' is not defined. "
                "Please check your waveform expression is correct."
            ) from ex
    return args

############################################################################
@dataclass
class DriveTermSymbolic:
    # --- Physics ---
    driven_op: qutip.Qobj
    modulation_freq: float         # GHz
    phi: float = 0.0               # rad phase

    # --- Envelope (provide ONE of the two, either as a function or as symbolic expression) ---
    pulse_shape_func: Optional[Callable] = None     # (t,args,math) -> float
    pulse_shape_args: Dict[str, float] = field(default_factory=dict)

    symbolic_expr: Optional[sp.Expr] = None         # SymPy ℰ(t)
    symbolic_params: Dict[sp.Symbol, float] = field(default_factory=dict)

    # --- Metadata ---
    pulse_id: Optional[str] = None
    qiskit_channel: str = "d0"
    dt: Optional[float] = 0.222 # Add dt for Qiskit sample conversion

    # internal lambdas
    _env_np: Optional[Callable] = field(init=False, default=None)
    _env_jx: Optional[Callable] = field(init=False, default=None)
    
    # Disable validation for JAX compatibility
    disable_validation: bool = False

    # ------------------------------------------------------------------
    def __post_init__(self):
        if self.symbolic_expr is not None:
            # substitute parameters → numeric expr
            t = sp.symbols("t")
            expr_with_params = self.symbolic_expr.subs(self.symbolic_params)
            
            # Create lambdified functions
            params = []
            for p in sorted(expr_with_params.free_symbols, key=lambda s: s.name):
                if p.name == "t":
                    params.insert(0, p)
                    continue
                params.append(p)
            
            # Create NumPy version
            self._env_np = sp.lambdify(params, [expr_with_params], modules=["numpy"])
            
            # Create JAX version if available
            if JAX_AVAILABLE:
                self._env_jx = sp.lambdify(params, [expr_with_params], modules=["jax"])
            
            # expose via a common callable API for QuTiP path
            self.pulse_shape_func = lambda tt, _=None, *, math=np: (
                self._env_np(tt) if math is np else self._env_jx(tt)
            )
            self.pulse_shape_args = {}
        elif self.pulse_shape_func is None:
            raise ValueError("DriveTerm needs either pulse_shape_func or symbolic_expr")

    # ------------------------------------------------------------
    # NumPy coefficient for QuTiP / Dynamiqs
    # ------------------------------------------------------------
    def numpy_coeff(self):
        cos = np.cos; two_pi = 2*np.pi; ω = self.modulation_freq; ϕ = self.phi
        return lambda t, args=None: (
            self.pulse_shape_func(t, self.pulse_shape_args, math=np)
            * cos(two_pi*ω*t - ϕ)
        )

    # JAX coefficient (optional for JAX-enabled QuTiP builds)
    def jax_coeff(self):
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX not installed")
        cos = jnp.cos; two_pi = 2*jnp.pi; ω = self.modulation_freq; ϕ = self.phi
        return lambda t, args=None: (
            self.pulse_shape_func(t, self.pulse_shape_args, math=jnp)
            * cos(two_pi*ω*t - ϕ)
        )

    # ------------------------------------------------------------
    # Qiskit Signals
    # ------------------------------------------------------------
    def to_qiskit_signal_numpy(self) -> Signal:
        if self.symbolic_expr is not None:
            env_np = self._env_np
        else:
            env_np = lambda t: self.pulse_shape_func(t, self.pulse_shape_args)
        return Signal(
            envelope=env_np,
            carrier_freq=self.modulation_freq,
            phase=self.phi,
            name=self.pulse_id,
        )

    def to_qiskit_signal_jax(self) -> Signal:
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX not installed")
        if self.symbolic_expr is not None:
            # Qiskit's ScalableSymbolicPulse expects an envelope expression with specific
            # placeholder symbols (usually _amp, _duration, _angle, and others for parameters).
            # It then substitutes the values passed to its constructor (amp, duration, angle)
            # and from its 'parameters' dict into this envelope.

            # 1. Define standard Qiskit placeholder symbols
            q_t = sp.symbols('t')
            q_amp = sp.symbols('amp')            # Placeholder for main amplitude
            q_duration = sp.symbols('duration')  # Placeholder for main duration
            q_angle = sp.symbols('angle')        # Placeholder for main angle (phase)

            # 2. Prepare substitution map and extract values for SSP constructor
            ssp_amp_val = None
            ssp_duration_val = None
            ssp_angle_val = 0.0  # Default, can be made configurable if needed
            ssp_parameters = {}
            subs_map = {sp.Symbol('t'): q_t} # Always map 't' to qiskit's 't'

            # Find our actual amp and t_duration symbols (defined in pulse_shapes_symbo)
            # This requires knowing their string names. A more robust way might be needed
            # if symbol names change, but for now, we assume standard names.
            our_amp_symbol = next((s for s in self.symbolic_params if str(s) == 'amp'), None)
            our_duration_symbol = next((s for s in self.symbolic_params if str(s) == 't_duration'), None)
            # Potentially, an angle symbol for the envelope could be sought here too.

            if our_amp_symbol:
                ssp_amp_val = self.symbolic_params[our_amp_symbol]
                subs_map[our_amp_symbol] = q_amp
            else:
                # If 'amp' is not in symbolic_params, SSP might still expect _amp in envelope.
                # For now, let's assume if not provided, it implies amp=1 in the template or it has to be there.
                # This case needs careful consideration based on how pulse_shapes are defined.
                # A simple solution: if 'amp' is a free symbol in symbolic_expr but not in params, it should map to q_amp.
                if sp.Symbol('amp') in self.symbolic_expr.free_symbols:
                    subs_map[sp.Symbol('amp')] = q_amp
                    if ssp_amp_val is None: ssp_amp_val = 1.0 # Default for SSP if not specified
                # Fallback if our_amp_symbol is None but we need a value
                if ssp_amp_val is None: ssp_amp_val = 1.0 

            if our_duration_symbol:
                ssp_duration_val = self.symbolic_params[our_duration_symbol]
                subs_map[our_duration_symbol] = q_duration # Map our t_duration to qiskit's duration placeholder
            else:
                # If t_duration is not directly in symbolic_params, ScalableSymbolicPulse still needs a duration.
                # This path should ideally not be hit if test_drive_symbo provides t_duration.
                # For safety, if a symbol named 'duration' (Qiskit style) is in the expression and not yet mapped:
                if sp.Symbol('duration') in self.symbolic_expr.free_symbols and sp.Symbol('duration') not in subs_map:
                    subs_map[sp.Symbol('duration')] = q_duration # Map it to placeholder
                    # Try to get its value if it was in symbolic_params under the key sp.Symbol('duration')
                    if sp.Symbol('duration') in self.symbolic_params:
                         ssp_duration_val = self.symbolic_params[sp.Symbol('duration')]

            if ssp_duration_val is None:
                raise ValueError(
                    "Duration in seconds for ScalableSymbolicPulse could not be determined. "
                    "Ensure 't_duration' (as a sympy.Symbol from pulse_shapes_symbo) is in symbolic_params, "
                    "or a symbol named 'duration' is in symbolic_params."
                )
            
            ssp_duration_samples = int(round(ssp_duration_val / self.dt))
            if ssp_duration_samples <= 0:
                raise ValueError(f"Calculated pulse duration in samples ({ssp_duration_samples}) must be positive. "
                                 f"Duration: {ssp_duration_val}, dt: {self.dt}")

            # Handle other parameters: these go into ssp_parameters
            # and their symbols in the expression need to be mapped to new qiskit-style placeholders.
            for user_sym, value in self.symbolic_params.items():
                user_sym_name = str(user_sym)
                if user_sym not in subs_map: # if not already mapped (t, amp, t_duration)
                    q_param_sym = sp.Symbol(user_sym_name) # Qiskit placeholder uses the same name
                    subs_map[user_sym] = q_param_sym
                    ssp_parameters[user_sym_name] = value
            
            # Also map any remaining free symbols in the original expression that weren't in symbolic_params
            # These will become Qiskit placeholders too, and SSP will expect them in its `parameters` dict
            # or they must be `amp`, `duration`, `angle` which are special. This step is crucial.
            for free_sym in self.symbolic_expr.free_symbols:
                if free_sym not in subs_map and str(free_sym) != 't': # if not t and not already handled
                    # Decide if it maps to q_amp, q_duration, q_angle, or a new q_param_sym
                    fs_name = str(free_sym)
                    if fs_name == 'amp' and our_amp_symbol is None:
                        subs_map[free_sym] = q_amp
                        if ssp_amp_val is None: ssp_amp_val = 1.0 # Default for SSP
                    elif fs_name == 'duration' and our_duration_symbol is None: # Qiskit uses 'duration' not 't_duration'
                        subs_map[free_sym] = q_duration
                        # If this happens, ssp_duration_val should have been set or error raised
                    elif fs_name == 'angle': # Qiskit uses 'angle'
                        subs_map[free_sym] = q_angle
                        # ssp_angle_val is already 0.0, could fetch from symbolic_params if an 'angle' symbol exists
                    else: # It's some other parameter
                        q_param_sym = sp.Symbol(fs_name)
                        subs_map[free_sym] = q_param_sym
                        # If this symbol wasn't in self.symbolic_params, its value for SSP is undefined.
                        # SSP would error if it expects a value for this in its `parameters` dict and doesn't find it.
                        # For now, we assume all necessary values are in self.symbolic_params.
                        if fs_name not in ssp_parameters and fs_name not in ['amp', 'duration', 'angle']:
                            # This implies a symbol is in the expression but not in symbolic_params
                            # This should ideally be caught by validation or __post_init__ for other paths.
                            # For SSP, it might mean it has a default or is an error with SSP itself.
                            pass # Let SSP handle it, or error if value missing

            # 3. Create the envelope expression for ScalableSymbolicPulse
            ssp_envelope_expr = self.symbolic_expr.subs(subs_map)

            # Create a ScalableSymbolicPulse with our expression
            this_pulse = pulse.ScalableSymbolicPulse(
                pulse_type=self.pulse_id or "custom_symbolic",
                duration=ssp_duration_samples, # Use duration in samples
                amp=ssp_amp_val,
                angle=ssp_angle_val, 
                envelope=ssp_envelope_expr,
                parameters=ssp_parameters,
                # TODO: Add constraints and valid_amp_conditions if available/needed
            )
            # Disable validation for JAX compatibility if needed (Qiskit versions vary)
            # pulse.ScalableSymbolicPulse.disable_validation = True 
            # Check if this attribute exists before setting, as it might be version-specific
            if hasattr(pulse.ScalableSymbolicPulse, 'disable_validation'):
                pulse.ScalableSymbolicPulse.disable_validation = True
            
            # build a pulse schedule
            with pulse.build() as schedule:
                pulse.play(this_pulse, pulse.DriveChannel(0))
            
            # Convert to signal
            # Ensure qiskit_channel is used correctly, assuming it maps to an integer for DriveChannel index if needed,
            # or directly as a string for the carriers dictionary key.
            # For DriveChannel, it expects an integer index. We need a mapping if qiskit_channel is like "d0".
            # For simplicity, if qiskit_channel is "d0", use 0. This might need a more robust mapping.
            channel_index = 0 # Default
            if isinstance(self.qiskit_channel, str) and self.qiskit_channel.startswith('d'):
                try:
                    channel_index = int(self.qiskit_channel[1:])
                except ValueError:
                    print(f"Warning: Could not parse channel index from qiskit_channel '{self.qiskit_channel}'. Defaulting to 0.")
            elif isinstance(self.qiskit_channel, int):
                channel_index = self.qiskit_channel

            with pulse.build() as schedule_final:
                pulse.play(this_pulse, pulse.DriveChannel(channel_index))
            
            converter = InstructionToSignals(dt=self.dt, carriers={self.qiskit_channel: self.modulation_freq})
            return converter.get_signals(schedule_final)[0]
        else:
            raise ValueError("Symbolic expression is not available, cannot convert to Qiskit Signal via ScalableSymbolicPulse")
