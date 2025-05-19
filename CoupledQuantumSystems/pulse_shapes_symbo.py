############################################################################
# pulse_shapes_symbo.py – symbolic pulse shapes
# ---------------------------------------------------------------------------
# This file contains symbolic definitions of pulse shapes using SymPy.
# These can be used with the DriveTerm class from drive_symbo.py.
############################################################################
from __future__ import annotations

import sympy as sp
from typing import Dict, Tuple, List, Optional, Set, Callable, Any
from dataclasses import dataclass, field
import inspect
import numpy as np # For PulseShapeCallable fallback

# Special symbols used by Qiskit's ScalableSymbolicPulse
QISKIT_SPECIAL_SYMBOLS = {'t', 'duration', 'amp', 'angle'}

# Common symbols used across pulse shapes
t_sym = sp.symbols('t', real=True)
# amp = sp.symbols('amp', real=True) # Amplitude will be handled by DriveTermSymbolic
start = sp.symbols('start', real=True)
length = sp.symbols('length', real=True)
rise = sp.symbols('rise', real=True)
square = sp.symbols('square', real=True)
stop = sp.symbols('stop', real=True)
amp_correction = sp.symbols('amp_correction', real=True) # For DRAG, relative coefficient
how_many_sigma = sp.symbols('how_many_sigma', real=True)
stoke = sp.symbols('stoke', real=True) # Boolean or indicator for STIRAP
delta1 = sp.symbols('delta1', real=True)
delta2 = sp.symbols('delta2', real=True)
# phi = sp.symbols('phi', real=True) # Envelope phase will be handled by DriveTermSymbolic
normalize = sp.symbols('normalize', real=True) # Boolean

@dataclass
class PulseParameters:
    """Container for pulse parameters with documentation and validation.
    
    This class ensures that:
    1. All required parameters are actually used in the expression
    2. All optional parameters with defaults are actually used in the expression
    3. No extra parameters are provided that aren't used
    4. Clear error messages are provided when validation fails
    
    Note: The following symbols are treated specially and don't need to be documented:
    - 't': The time variable (or t_sym)
    The following are handled by DriveTermSymbolic directly:
    - 'amp': Pulse amplitude
    - 'duration': Overall pulse duration for Qiskit
    - 'angle': Overall envelope phase for Qiskit
    """
    required_params: List[str]
    optional_params: Dict[str, float] = field(default_factory=dict)
    docstring: str = ""
    
    def validate_against_expr(self, expr: sp.Expr) -> None:
        """Validate that the parameters match the symbolic expression.
        
        Args:
            expr: The symbolic expression to validate against
            
        Raises:
            ValueError: If validation fails, with a clear error message
        """
        # Get all symbols used in the expression by their string name
        symbols_in_expr_str = {str(sym) for sym in expr.free_symbols}
        
        # Check required parameters: they must be present in the expression's free symbols.
        missing_required = set(self.required_params) - symbols_in_expr_str
        if missing_required:
            raise ValueError(
                f"Required parameters {sorted(list(missing_required))} are not used in the symbolic expression. "
                f"Symbols found in expression: {sorted(list(symbols_in_expr_str))}. "
                f"Expression: {expr}. "
                f"Please ensure these required parameters are part of the expression, or remove them from required_params."
            )
        
        symbols_needing_check = symbols_in_expr_str - {t_sym.name}
        documented_as_param_by_user = set(self.required_params) | set(self.optional_params.keys())
        
        other_qiskit_special_symbols = QISKIT_SPECIAL_SYMBOLS - {t_sym.name, 'amp', 'duration', 'angle'}
        
        extra_symbols = symbols_needing_check - documented_as_param_by_user - other_qiskit_special_symbols
        
        if extra_symbols:
            externalized_attrs = {'amp', 'angle'}
            improperly_used_external = extra_symbols & externalized_attrs
            if improperly_used_external:
                raise ValueError(
                    f"Symbols {sorted(list(improperly_used_external))} (e.g., 'amp', 'angle') are used in the expression "
                    f"but should be handled by DriveTermSymbolic directly, not as shape parameters. "
                    f"Expression: {expr}."
                )

            raise ValueError(
                f"Symbols {sorted(list(extra_symbols))} are used in the expression but not documented as required or optional parameters. "
                f"Documented parameters by user: {sorted(list(documented_as_param_by_user))}. "
                f"Expression: {expr}. "
                f"Please add them to required_params or optional_params, or ensure they are standard Qiskit symbols if intended."
            )

def validate_pulse_definition(func):
    """Decorator to validate pulse definitions.
    
    This decorator ensures that:
    1. The function returns a tuple of (expr, defaults, params)
    2. The params match the expression (excluding 't' and primary DriveTermSymbolic attributes like 'amp', 'duration', 'angle')
    3. The defaults match the optional_params
    """
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if not isinstance(result, tuple) or len(result) != 3:
            raise ValueError(
                f"Pulse definition {func.__name__} must return (expr, defaults, params). "
                f"Got {type(result)} with length {len(result) if isinstance(result, tuple) else 'N/A'}"
            )
        
        expr, defaults, params_obj = result # Renamed params to params_obj to avoid conflict
        if not isinstance(params_obj, PulseParameters):
            raise ValueError(
                f"Pulse definition {func.__name__} must return PulseParameters as third element. "
                f"Got {type(params_obj)}"
            )
        
        params_obj.validate_against_expr(expr)
        
        default_symbols_names = {str(sym_obj.name) for sym_obj in defaults.keys()
                                 if str(sym_obj.name) not in (QISKIT_SPECIAL_SYMBOLS | {'amp', 'angle'})}
        optional_params_names = set(params_obj.optional_params.keys())
        
        # All keys in defaults (their names) MUST be in optional_params_names
        missing_in_optional = default_symbols_names - optional_params_names
        if missing_in_optional:
            raise ValueError(
                f"Defaults in {func.__name__} contains keys {missing_in_optional} not in optional_params. "
                f"Default symbol names: {default_symbols_names}, Optional param names: {optional_params_names}"
            )
        # All optional_params_names should ideally have a default if not required.
        # Ensure default_symbols_names is a subset of optional_params_names after filtering.
        # This check is subtly different from before, focusing on names.
        if not default_symbols_names.issubset(optional_params_names):
             raise ValueError(
                f"Defaults keys in {func.__name__} (after filtering special symbols) must be a subset of optional_params keys. "
                f"Filtered default symbol names: {default_symbols_names}, Optional param names: {optional_params_names}"
            )
        return result
    return wrapper

@validate_pulse_definition
def square_pulse_with_rise_fall() -> Tuple[sp.Expr, Dict[sp.Symbol, float], PulseParameters]:
    """Symbolic definition of a square pulse envelope with rise and fall times.
    The envelope is unscaled (peak amplitude of 1 for the square part)."""
    t_fall_start = start + rise + square
    end_time = t_fall_start + rise # Renamed 'end' to 'end_time' to avoid conflict
    
    rise_window = sp.Piecewise(
        (1, (t_sym >= start) & (t_sym <= start + rise)),
        (0, True)
    )
    square_window_val = sp.Piecewise( # Renamed 'square_window' to 'square_window_val'
        (1, (t_sym >= start + rise) & (t_sym <= t_fall_start)),
        (0, True)
    )
    fall_window = sp.Piecewise(
        (1, (t_sym >= t_fall_start) & (t_sym <= end_time)),
        (0, True)
    )
    
    rise_envelope = rise_window * sp.sin(sp.pi * (t_sym - start) / (2 * rise)) ** 2
    square_envelope_val = square_window_val # Renamed 'square_envelope' to 'square_envelope_val'
    fall_envelope = fall_window * sp.sin(sp.pi * (end_time - t_sym) / (2 * rise)) ** 2
    
    envelope = (square_envelope_val + rise_envelope + fall_envelope)
    
    defaults = {
        start: 0.0,
        rise: 1e-13,
        square: 0.0
    }
    
    params = PulseParameters(
        required_params=[], # No 'amp'
        optional_params={
            'start': 0.0,
            'rise': 1e-13,
            'square': 0.0
        },
        docstring="""Square pulse shape with rise and fall times. Peak is 1.
            
        Optional parameters:
        -------------------
        start: float
            Start time of the pulse (default: 0)
        rise: float
            Rise time in seconds (default: 1e-13). Must be > 0 if used.
        square: float
            Duration of constant amplitude (default: 0)
        """
    )
    
    return envelope, defaults, params

@validate_pulse_definition
def sin_squared_pulse() -> Tuple[sp.Expr, Dict[sp.Symbol, float], PulseParameters]:
    """Symbolic definition of a sin-squared pulse envelope.
    The envelope is unscaled (peak amplitude of 1)."""
    end_time = start + length # Renamed 'end' to 'end_time'
    
    inside_window = sp.Piecewise(
        (1, (t_sym >= start) & (t_sym <= end_time)),
        (0, True)
    )
    
    envelope = inside_window * sp.sin(sp.pi * (t_sym - start) / length) ** 2
    
    defaults = {
        start: 0.0
    }
    
    params = PulseParameters(
        required_params=['length'], # No 'amp'
        optional_params={
            'start': 0.0
        },
        docstring="""Sin-squared pulse shape. Peak is 1.
        
        Required parameters:
        -------------------
        length: float
            Duration of the pulse in seconds. Must be > 0.
            
        Optional parameters:
        -------------------
        start: float
            Start time of the pulse (default: 0)
        """
    )
    
    return envelope, defaults, params

@validate_pulse_definition
def sin_squared_DRAG() -> Tuple[sp.Expr, Dict[sp.Symbol, float], PulseParameters]:
    """Symbolic definition of a sin-squared DRAG pulse envelope.
    The main envelope component is unscaled (peak amplitude of 1).
    'amp_correction' is the coefficient for the derivative term."""
    end_time = start + length # Renamed 'end' to 'end_time'
    
    inside_window = sp.Piecewise(
        (1, (t_sym >= start) & (t_sym <= end_time)),
        (0, True)
    )
    
    main_envelope = inside_window * sp.sin(sp.pi * (t_sym - start) / length) ** 2
    derivative_envelope = inside_window * (sp.pi/length) * sp.sin(2 * sp.pi * (t_sym - start) / length)
    
    # amp_correction is the coefficient beta in E(t) * (1 - i * beta * E_dot(t) / E(t))
    # Here, E(t) is main_envelope, E_dot(t) is derivative_envelope.
    # If ScalableSymbolicPulse applies A * exp(i*phi) * envelope_expr,
    # and envelope_expr is (main_envelope - I * amp_correction * derivative_envelope),
    # then amp_correction is indeed the beta.
    complex_envelope = (main_envelope - sp.I * amp_correction * derivative_envelope)
    
    defaults = {
        start: 0.0
    }
    
    params = PulseParameters(
        required_params=['length', 'amp_correction'], # No 'amp'
        optional_params={
            'start': 0.0
        },
        docstring="""Sin-squared DRAG pulse shape. Main component peak is 1.
        
        Required parameters:
        -------------------
        length: float
            Duration of the pulse in seconds. Must be > 0.
        amp_correction: float
            DRAG coefficient (beta) for the derivative term.
            
        Optional parameters:
        -------------------
        start: float
            Start time of the pulse (default: 0)
        """
    )
    
    return complex_envelope, defaults, params

@validate_pulse_definition
def gaussian_pulse() -> Tuple[sp.Expr, Dict[sp.Symbol, float], PulseParameters]:
    """Symbolic definition of a Gaussian pulse envelope.
    Unscaled before normalization (based on exp function).
    If normalized=True, peak is 1 (after windowing and shifting)."""
    sigma = length / how_many_sigma
    t_center = start + length / 2
    
    # Pure Gaussian shape, peak is 1 at t_center if not windowed
    gaussian_shape = sp.exp(-((t_sym - t_center) ** 2) / (2 * sigma ** 2))
    
    end_time = start + length # Renamed 'end' to 'end_time'
    inside_window = sp.Piecewise(
        (1, (t_sym >= start) & (t_sym <= end_time)),
        (0, True)
    )
    
    windowed_gaussian = inside_window * gaussian_shape
    
    # Normalization logic (lifted Gaussian)
    val_at_start = gaussian_shape.subs(t_sym, start) # Value of pure shape at window start
    
    # If normalize=True, make it (G(t)-G(start))/(1-G(start)) within window
    # If normalize=False, just use G(t) within window
    # The '1' in (1-a) assumes peak of pure gaussian is 1.
    # For G(t) = exp(-...), max is 1. G(start) < 1. So (1-a) is positive.
    envelope = sp.Piecewise(
        (((windowed_gaussian - val_at_start * inside_window) / (1 - val_at_start)), normalize),
        (windowed_gaussian, True) # Default to non-normalized if normalize is False
    )
        
    defaults = {
        start: 0.0,
        how_many_sigma: 6.0,
        normalize: False 
    }
    
    params = PulseParameters(
        required_params=['length'], # No 'amp'
        optional_params={
            'start': 0.0,
            'how_many_sigma': 6.0,
            'normalize': False
        },
        docstring="""Gaussian pulse shape.
        If normalize=False, it's a windowed Gaussian (peak of exp is 1).
        If normalize=True, it's a lifted Gaussian (starts at 0, peak is 1).
        
        Required parameters:
        -------------------
        length: float
            Duration of the pulse window in seconds. Sigma is derived from this. Must be > 0.
            
        Optional parameters:
        -------------------
        start: float
            Start time of the pulse window (default: 0)
        how_many_sigma: float
            Defines sigma = length / how_many_sigma (default: 6.0). Must be > 0.
        normalize: bool
            Whether to apply lifted normalization (default: False).
        """
    )
    
    return envelope, defaults, params

@validate_pulse_definition
def gaussian_DRAG() -> Tuple[sp.Expr, Dict[sp.Symbol, float], PulseParameters]:
    """Symbolic definition of a Gaussian DRAG pulse envelope.
    Main Gaussian component unscaled/normalized as in gaussian_pulse.
    'amp_correction' is the coefficient for the derivative term."""
    sigma = length / how_many_sigma
    t_center = start + length / 2
    
    gaussian_shape = sp.exp(-((t_sym - t_center) ** 2) / (2 * sigma ** 2))
    
    end_time = start + length # Renamed 'end' to 'end_time'
    inside_window = sp.Piecewise(
        (1, (t_sym >= start) & (t_sym <= end_time)),
        (0, True)
    )
    windowed_gaussian = inside_window * gaussian_shape
    val_at_start = gaussian_shape.subs(t_sym, start)
    
    main_envelope = sp.Piecewise(
        (((windowed_gaussian - val_at_start * inside_window) / (1 - val_at_start)), normalize),
        (windowed_gaussian, True)
    )
    
    # Derivative of the pure (unwindowed, unnormalized) Gaussian shape for DRAG
    # d/dt exp(- (t-c)^2 / (2s^2) ) = exp(- (t-c)^2 / (2s^2) ) * (-(t-c)/s^2)
    # The derivative should be of the main_envelope that is actually used.
    # If main_envelope is normalized, derivative of normalized form is complex.
    # Standard DRAG: Envelope_I + i * beta * d/dt(Envelope_I)
    # Let Envelope_I be main_envelope. We need its derivative.
    # For simplicity, Qiskit often uses derivative of symbolic pulse.
    # Let's use derivative of the 'windowed_gaussian' before normalization for DRAG part,
    # as amp_correction is often calibrated against the fundamental Gaussian parameters.
    # derivative_term_shape = gaussian_shape * (-(t - t_center)/sigma**2) # Derivative of pure gaussian
    # windowed_derivative = inside_window * derivative_term_shape
    # This is tricky: if main_envelope is normalized, derivative part should also be consistent.
    # Let's assume amp_correction applies to derivative of the *final* main_envelope.
    # Sympy can compute derivative of main_envelope directly.
    
    derivative_of_main_envelope = sp.diff(main_envelope, t_sym)

    # The drag_correction from the original user code was:
    # drag_correction = 1 + sp.I * amp_correction * (-(t - t_center)/sigma**2)
    # This implies the Q component is amp_correction * (-(t-c)/s^2) * main_gaussian_component
    # which is amp_correction * (-1/amp) * d/dt(amp * gaussian_shape) if amp was part of gaussian_shape.
    # Now that gaussian_shape is pure, derivative_of_pure_gaussian = gaussian_shape * (-(t-c)/s^2)
    # So Q component: amp_correction * derivative_of_pure_gaussian * inside_window (if not normalized)
    # This is more standard for DRAG where correction is related to anharm and sigma.
    
    # Let's use the common DRAG form: G(t) + i * beta * G_dot(t)
    # where G(t) is the main_envelope (potentially normalized)
    # and G_dot(t) is its derivative. 'amp_correction' is beta.
    
    complex_envelope = main_envelope + sp.I * amp_correction * derivative_of_main_envelope
    
    defaults = {
        start: 0.0,
        how_many_sigma: 6.0,
        normalize: False
    }
    
    params = PulseParameters(
        required_params=['length', 'amp_correction'], # No 'amp'
        optional_params={
            'start': 0.0,
            'how_many_sigma': 6.0,
            'normalize': False
        },
        docstring="""Gaussian DRAG pulse shape.
        Main Gaussian component as per gaussian_pulse.
        'amp_correction' is the DRAG coefficient (beta).
        
        Required parameters:
        -------------------
        length: float
            Duration of the pulse window. Sigma derived from this. Must be > 0.
        amp_correction: float
            DRAG coefficient (beta).
            
        Optional parameters:
        -------------------
        start: float
            Start time of the pulse window (default: 0)
        how_many_sigma: float
            Defines sigma = length / how_many_sigma (default: 6.0). Must be > 0.
        normalize: bool
            Whether to apply lifted normalization to main Gaussian component (default: False).
        """
    )
    
    return complex_envelope, defaults, params

@validate_pulse_definition
def STIRAP_stoke() -> Tuple[sp.Expr, Dict[sp.Symbol, float], PulseParameters]:
    """Symbolic definition of a STIRAP stoke pulse envelope.
    Unscaled (based on exp and trig functions, normalized form)."""
    lambda_val = sp.Float(4.0) # Ensure sympy float
    tau_for_mono = (stop - start) / sp.Float(6.0)
    center = (stop - start) / sp.Float(2.0) + start
    
    # mono_increasing can be negative if t < center for exp argument
    mono_increasing = sp.Function('mono_increasing')
    try:
        mono_increasing_expr = 1 / (1 + sp.exp(-lambda_val * (t_sym - center) / tau_for_mono))
    except OverflowError: # Should not happen with symbolic
        mono_increasing_expr = sp.Rational(1,2) # Fallback, though symbolic should handle

    hyper_Gaussian_exponent = -((t_sym - center) / (2 * tau_for_mono)) ** sp.Integer(6)
    hyper_Gaussian = sp.exp(hyper_Gaussian_exponent)
    
    val_at_start = hyper_Gaussian.subs(t_sym, start) # Value of pure shape at window start
    
    # Normalized hyper-Gaussian part
    # Ensure (1-a) is not zero; for hypergaussian this should be fine if stop > start
    # Using sp.Max to avoid division by zero if val_at_start is very close to 1 (though unlikely for typical STIRAP)
    #denominator = sp.Max(1 - val_at_start, 1e-9) # Avoid instability if val_at_start is 1
    # A simpler assumption: 1-val_at_start won't be zero for valid params.
    normalized_hyper_G = (hyper_Gaussian - val_at_start) / (1 - val_at_start)
    
    # Stoke pulse uses cos part of the mixing angle
    # envelope = normalized_hyper_G * sp.cos(sp.pi/2 * mono_increasing_expr) # old symbolic function call
    envelope = normalized_hyper_G * sp.cos(sp.pi/2 * (1 / (1 + sp.exp(-lambda_val * (t_sym - center) / tau_for_mono))))


    defaults = {
        start: 0.0
    }
    
    params = PulseParameters(
        required_params=['stop'], # No 'amp'
        optional_params={
            'start': 0.0
        },
        docstring="""STIRAP stoke pulse shape (normalized).
        Based on hyper-Gaussian and sinusoidal mixing angle.
        
        Required parameters:
        -------------------
        stop: float
            Stop time of the pulse in seconds. Must be > start.
            
        Optional parameters:
        -------------------
        start: float
            Start time of the pulse (default: 0)
        """
    )
    
    return envelope, defaults, params

@validate_pulse_definition
def STIRAP_pump() -> Tuple[sp.Expr, Dict[sp.Symbol, float], PulseParameters]:
    """Symbolic definition of a STIRAP pump pulse envelope.
    Unscaled (based on exp and trig functions, normalized form)."""
    lambda_val = sp.Float(4.0)
    tau_for_mono = (stop - start) / sp.Float(6.0)
    center = (stop - start) / sp.Float(2.0) + start
    
    # mono_increasing_expr = 1 / (1 + sp.exp(-lambda_val * (t_sym - center) / tau_for_mono)) # Re-inline
    
    hyper_Gaussian_exponent = -((t_sym - center) / (2 * tau_for_mono)) ** sp.Integer(6)
    hyper_Gaussian = sp.exp(hyper_Gaussian_exponent)
    
    val_at_start = hyper_Gaussian.subs(t_sym, start)
    # denominator = sp.Max(1 - val_at_start, 1e-9)
    normalized_hyper_G = (hyper_Gaussian - val_at_start) / (1 - val_at_start)

    # Pump pulse uses sin part of the mixing angle
    envelope = normalized_hyper_G * sp.sin(sp.pi/2 * (1 / (1 + sp.exp(-lambda_val * (t_sym - center) / tau_for_mono))))
    
    defaults = {
        start: 0.0
    }
    
    params = PulseParameters(
        required_params=['stop'], # No 'amp'
        optional_params={
            'start': 0.0
        },
        docstring="""STIRAP pump pulse shape (normalized).
        Based on hyper-Gaussian and sinusoidal mixing angle.
        
        Required parameters:
        -------------------
        stop: float
            Stop time of the pulse in seconds. Must be > start.
            
        Optional parameters:
        -------------------
        start: float
            Start time of the pulse (default: 0)
        """
    )
    
    return envelope, defaults, params

@validate_pulse_definition
def Hyper_Gaussian_DRAG_STIRAP_stoke() -> Tuple[sp.Expr, Dict[sp.Symbol, float], PulseParameters]:
    """Symbolic definition of a Hyper-Gaussian DRAG STIRAP stoke pulse.
    Main STIRAP stoke component is unscaled/normalized.
    'amp_correction' is the coefficient for the derivative term."""
    lambda_val = sp.Float(4.0)
    tau_mono = (stop - start) / sp.Float(6.0)
    center_time = (stop + start) / sp.Float(2.0) # Renamed center to center_time
        
    mono_expr = 1 / (1 + sp.exp(-lambda_val * (t_sym - center_time) / tau_mono))
    
    T0 = 2 * tau_mono
    hyperG_expr = sp.exp(-((t_sym - center_time) / T0) ** 6)
    # d_hyperG = hyperG_expr * (-6) * ((t - center_time) / T0) ** 5 / T0 # For derivative calc later
    
    a0 = hyperG_expr.subs(t_sym, start)
    
    # Main STIRAP stoke envelope (normalized)
    main_envelope = (hyperG_expr - a0) / (1 - a0) * sp.cos(sp.pi/2 * mono_expr)
    
    # Derivative of the main envelope for DRAG
    # d_mono = (lambda_val / tau_mono) * mono_expr * (1 - mono_expr)
    # d_main_envelope_term1 = (d_hyperG / (1 - a0) - (hyperG_expr - a0) / (1 - a0)**2 * d_hyperG.subs(t, start)) * sp.cos(sp.pi/2 * mono_expr)
    # d_main_envelope_term2 = (hyperG_expr - a0)/(1 - a0) * (-sp.pi/2) * sp.sin(sp.pi/2 * mono_expr) * d_mono
    # derivative_of_main_envelope = d_main_envelope_term1 + d_main_envelope_term2
    derivative_of_main_envelope = sp.diff(main_envelope, t_sym)

    complex_envelope = main_envelope - sp.I * amp_correction * derivative_of_main_envelope
    
    defaults = {
        start: 0.0
    }
    
    params = PulseParameters(
        required_params=['amp_correction', 'stop'], # No 'amp'
        optional_params={
            'start': 0.0
        },
        docstring="""Hyper-Gaussian DRAG STIRAP stoke pulse shape.
        Main stoke component is normalized. 'amp_correction' is DRAG coefficient.
        
        Required parameters:
        -------------------
        amp_correction: float
            DRAG coefficient (beta).
        stop: float
            Stop time of the pulse. Must be > start.
            
        Optional parameters:
        -------------------
        start: float
            Start time of the pulse (default: 0)
        """
    )
    
    return complex_envelope, defaults, params

@validate_pulse_definition
def Hyper_Gaussian_DRAG_STIRAP_pump() -> Tuple[sp.Expr, Dict[sp.Symbol, float], PulseParameters]:
    """Symbolic definition of a Hyper-Gaussian DRAG STIRAP pump pulse.
    Main STIRAP pump component is unscaled/normalized.
    'amp_correction' is the coefficient for the derivative term."""
    lambda_val = sp.Float(4.0)
    tau_mono = (stop - start) / sp.Float(6.0)
    center_time = (stop + start) / sp.Float(2.0) # Renamed center to center_time
    
    mono_expr = 1 / (1 + sp.exp(-lambda_val * (t_sym - center_time) / tau_mono))
    
    T0 = 2 * tau_mono
    hyperG_expr = sp.exp(-((t_sym - center_time) / T0) ** 6)
    # d_hyperG = hyperG_expr * (-6) * ((t - center_time) / T0) ** 5 / T0
    
    a0 = hyperG_expr.subs(t_sym, start)
    
    main_envelope = (hyperG_expr - a0) / (1 - a0) * sp.sin(sp.pi/2 * mono_expr)
    # derivative_of_main_envelope = (d_hyperG / (1 - a0) - (hyperG_expr - a0) / (1 - a0)**2 * d_hyperG.subs(t, start)) * sp.sin(sp.pi/2 * mono_expr) + \
    #              (hyperG_expr - a0)/(1 - a0) * (sp.pi/2) * sp.cos(sp.pi/2 * mono_expr) * (lambda_val / tau_mono) * mono_expr * (1 - mono_expr)
    derivative_of_main_envelope = sp.diff(main_envelope, t_sym)
    
    complex_envelope = main_envelope - sp.I * amp_correction * derivative_of_main_envelope
    
    defaults = {
        start: 0.0
    }
    
    params = PulseParameters(
        required_params=['amp_correction', 'stop'], # No 'amp'
        optional_params={
            'start': 0.0
        },
        docstring="""Hyper-Gaussian DRAG STIRAP pump pulse shape.
        Main pump component is normalized. 'amp_correction' is DRAG coefficient.
        
        Required parameters:
        -------------------
        amp_correction: float
            DRAG coefficient (beta).
        stop: float
            Stop time of the pulse. Must be > start.
            
        Optional parameters:
        -------------------
        start: float
            Start time of the pulse (default: 0)
        """
    )
    
    return complex_envelope, defaults, params

@validate_pulse_definition
def sin_squared_recursive_DRAG() -> Tuple[sp.Expr, Dict[sp.Symbol, float], PulseParameters]:
    """Symbolic definition of a recursive DRAG pulse envelope (unscaled).
    The main sin_squared component is unscaled (peak 1).
    Delta parameters define normalized DRAG coefficients."""
    # Normalized DRAG coefficients (kappa1/amp and kappa2/amp from original)
    # kappa2_norm = -1 / (delta1 * delta2)
    # kappa1_norm = (delta1 + delta2) / (delta1 * delta2)
    # For safety with symbolic division if delta1 or delta2 can be zero (though physically unlikely)
    # we should ensure they are non-zero or handle it. Assuming they are non-zero from context.
    kappa2_norm = -1 / (delta1 * delta2)
    kappa1_norm = (delta1 + delta2) / (delta1 * delta2)

    end_time = start + length # Renamed 'end' to 'end_time'
    two_pi_over_L = 2 * sp.pi / length # Renamed T to L for length
    pi_over_L = sp.pi / length
    
    inside = sp.Piecewise(
        (1, (t_sym >= start) & (t_sym <= end_time)),
        (0, True)
    )
    
    # Main sin^2 envelope, peak is 1
    main_envelope = inside * sp.sin(sp.pi * (t_sym - start) / length) ** 2
    
    # 1st derivative of main_envelope
    # derivative1 = inside * pi_over_L * sp.sin(two_pi_over_L * (t - start))
    derivative1 = sp.diff(main_envelope, t_sym) # More robust way to get derivative
    
    # 2nd derivative of main_envelope
    # derivative2 = inside * 2 * (pi_over_L ** 2) * (sp.cos(two_pi_over_L * (t - start)) - 1)
    # Sympy might simplify the derivative of Piecewise, which is good.
    # For recursive DRAG, often derivatives of the un-windowed shape are used, then windowed.
    # Let's use derivative of the windowed main_envelope.
    derivative2 = sp.diff(derivative1, t_sym)
        
    baseband_envelope = main_envelope + sp.I * kappa1_norm * derivative1 + kappa2_norm * derivative2
    
    defaults = {
        start: 0.0
    }
    
    params = PulseParameters(
        required_params=['delta1', 'delta2', 'length'], # No 'amp'
        optional_params={
            'start': 0.0
        },
        docstring="""Recursive DRAG pulse shape with two spectral notches.
        Main sin_squared component has peak 1.
        
        Required parameters:
        -------------------
        delta1: float
            Detuning Δ₁ (rad/s) of 1st unwanted transition. Must be non-zero.
        delta2: float
            Detuning Δ₂ (rad/s) of 2nd unwanted transition. Must be non-zero.
        length: float
            Pulse length in seconds. Must be > 0.
            
        Optional parameters:
        -------------------
        start: float
            Left edge of pulse window (default: 0)
        """
    )
    
    return baseband_envelope, defaults, params 

# --- Pulse Registry and Creation Logic (Simplified) ---

@dataclass
class PulseSpecEntry:
    """Holds the processed information for a pulse type."""
    pure_expr: sp.Expr
    ordered_param_symbols: List[sp.Symbol]
    t_symbol: sp.Symbol
    validator_params_obj: PulseParameters # For validating inputs in create_pulse_shape
    # defaults: Dict[sp.Symbol, float] # Kept by validator_params_obj.optional_params

PULSE_PARAM_SPECS: Dict[str, PulseSpecEntry] = {}

def _get_global_symbol_by_name(name: str, default_if_not_found: Optional[sp.Symbol] = None) -> Optional[sp.Symbol]:
    """Helper to get a globally defined sympy symbol object by its string name."""
    # Check common symbols defined in this file first
    known_symbols = {
        't': t_sym, 'start': start, 'length': length, 'rise': rise, 'square': square,
        'stop': stop, 'amp_correction': amp_correction, 'how_many_sigma': how_many_sigma,
        'stoke': stoke, 'delta1': delta1, 'delta2': delta2, 'normalize': normalize
    }
    if name in known_symbols:
        return known_symbols[name]
    
    # Fallback to checking globals(), though less robust if symbol isn\'t directly in module globals
    sym_obj = globals().get(name)
    if isinstance(sym_obj, sp.Symbol):
        return sym_obj
    return default_if_not_found

def _register_pulse_spec(
    name: str, 
    func: Callable[[], Tuple[sp.Expr, Dict[sp.Symbol, float], PulseParameters]]
):
    expr, defaults, params_obj = func() # Call the @validated pulse definition function

    # Determine ordered_shape_param_symbols based on params_obj and expr.free_symbols
    # The order is: required params (in order of definition), then optional params (alphabetical)
    ordered_symbols_list: List[sp.Symbol] = []
    processed_names: Set[str] = set()

    # 1. Add required parameters in their defined order
    for req_name in params_obj.required_params:
        sym_obj = _get_global_symbol_by_name(req_name)
        if sym_obj is None:
            # This case implies a required parameter was named but not defined as a global sympy symbol
            # Or _get_global_symbol_by_name couldn\'t find it.
            # For robustness, one might create sp.symbols(req_name) here, but it\'s better to predefine all.
            raise ValueError(f"Symbol for required parameter \'{req_name}\' not found for pulse \'{name}\'. Ensure it\'s a predefined global symbol.")
        if sym_obj not in expr.free_symbols and sym_obj != t_sym:
             # This should ideally be caught by params_obj.validate_against_expr if sym_obj is not t_sym
             # print(f"Warning: Required param symbol {sym_obj} for {name} not in expression free_symbols {expr.free_symbols}")
             pass # Allow, as validate_against_expr should have caught if its string name is missing.
        ordered_symbols_list.append(sym_obj)
        processed_names.add(req_name)

    # 2. Add optional parameters, sorted alphabetically, if they are in the expression
    optional_names_sorted = sorted(params_obj.optional_params.keys())
    for opt_name in optional_names_sorted:
        if opt_name not in processed_names:
            sym_obj = _get_global_symbol_by_name(opt_name)
            if sym_obj is None:
                raise ValueError(f"Symbol for optional parameter \'{opt_name}\' not found for pulse \'{name}\'. Ensure it\'s a predefined global symbol.")
            # Only add if it actually appears in the expression\'s free symbols
            if sym_obj in expr.free_symbols and sym_obj != t_sym:
                ordered_symbols_list.append(sym_obj)
                processed_names.add(opt_name)
            elif sym_obj in defaults: # If it has a default, it might be used implicitly
                # Still add it to the list if it has a defined symbol, as create_pulse_shape might need to pass default
                # However, for lambdify, only free symbols in the expr should be args.
                # For now, stick to free symbols for ordered_symbols_list for lambdify.
                pass 

    # Final check: ensure ordered_symbols_list only contains free symbols of the expression (excluding t_sym)
    final_ordered_symbols = []
    expr_free_params = expr.free_symbols - {t_sym}
    temp_map_for_final_order = {s.name: s for s in expr_free_params}

    # Use the previously determined order (req then opt sorted) but filter strictly by free_symbols
    for sym_in_initial_order in ordered_symbols_list:
        if sym_in_initial_order.name in temp_map_for_final_order:
            final_ordered_symbols.append(temp_map_for_final_order[sym_in_initial_order.name])
            del temp_map_for_final_order[sym_in_initial_order.name] # Avoid duplicates if logic error
    # Add any remaining free symbols that weren\'t in req/opt (should not happen if PulseParameters is correct)
    # Sort them for consistency if any exist
    for remaining_sym_name in sorted(temp_map_for_final_order.keys()):
        # print(f"Warning: Symbol {remaining_sym_name} in expression but not in ordered params for {name}. Appending.")
        final_ordered_symbols.append(temp_map_for_final_order[remaining_sym_name])


    spec_entry = PulseSpecEntry(
        pure_expr=expr,
        ordered_param_symbols=final_ordered_symbols,
        t_symbol=t_sym, # Assuming all use the global t_sym
        validator_params_obj=params_obj
    )
    PULSE_PARAM_SPECS[name] = spec_entry

# Register all defined pulse shapes
# These functions are defined above in the file and decorated with @validate_pulse_definition
_register_pulse_spec("square_pulse_with_rise_fall", square_pulse_with_rise_fall)
_register_pulse_spec("sin_squared_pulse", sin_squared_pulse)
_register_pulse_spec("sin_squared_DRAG", sin_squared_DRAG)
_register_pulse_spec("gaussian_pulse", gaussian_pulse)
_register_pulse_spec("gaussian_DRAG", gaussian_DRAG)
_register_pulse_spec("STIRAP_stoke", STIRAP_stoke)
_register_pulse_spec("STIRAP_pump", STIRAP_pump)
_register_pulse_spec("Hyper_Gaussian_DRAG_STIRAP_stoke", Hyper_Gaussian_DRAG_STIRAP_stoke)
_register_pulse_spec("Hyper_Gaussian_DRAG_STIRAP_pump", Hyper_Gaussian_DRAG_STIRAP_pump)
_register_pulse_spec("sin_squared_recursive_DRAG", sin_squared_recursive_DRAG)

def create_pulse_shape(
    pulse_type: str, 
    params_values: Dict[str, float] # User-provided: Dict[param_name_str, value_float]
) -> Tuple[sp.Expr, List[sp.Symbol], sp.Symbol]:
    """Retrieve essential components for a pulse shape.

    Args:
        pulse_type: The registered name of the pulse shape.
        params_values: Dictionary of parameter names (strings) to their float values.
                       These are for the shape-specific parameters.

    Returns:
        A tuple: (symbolic_expr_pure, ordered_shape_param_symbols, t_symbol_used)
        - symbolic_expr_pure: The pure SymPy expression for the shape.
        - ordered_shape_param_symbols: Ordered list of SymPy symbols for shape parameters.
        - t_symbol_used: The SymPy time symbol used in the expression.
    """
    if pulse_type not in PULSE_PARAM_SPECS:
        raise ValueError(f"Unknown pulse_type: \'{pulse_type}\'. Available: {list(PULSE_PARAM_SPECS.keys())}")

    spec = PULSE_PARAM_SPECS[pulse_type]
    validator = spec.validator_params_obj

    # Validate provided params_values against the pulse\'s parameter specification
    # 1. Check for missing required parameters
    for req_param_name in validator.required_params:
        if req_param_name not in params_values:
            raise ValueError(
                f"Missing required parameter \'{req_param_name}\' for pulse_type \'{pulse_type}\'. "
                f"Required: {validator.required_params}, Provided: {list(params_values.keys())}"
            )

    # 2. Check for unknown parameters
    allowed_param_names = set(validator.required_params) | set(validator.optional_params.keys())
    for provided_name in params_values.keys():
        if provided_name not in allowed_param_names:
            raise ValueError(
                f"Unknown parameter \'{provided_name}\' provided for pulse_type \'{pulse_type}\'. "
                f"Allowed: {sorted(list(allowed_param_names))}, Provided: {list(params_values.keys())}"
            )
    
    # All checks passed. Return the core components.
    # DriveTermSymbolic will use these to lambdify and manage the pulse.
    # Note: params_values provided to this function are string-keyed. 
    # DriveTermSymbolic.symbolic_params is Symbol-keyed. The mapping happens there.

    return spec.pure_expr, spec.ordered_param_symbols, spec.t_symbol

# Make sure the global `t_sym` is easily importable if needed elsewhere as `default_t_sym`
# (though `drive_symbo.py` already imports `t_sym as default_t_sym` if `t_sym` is global here) 