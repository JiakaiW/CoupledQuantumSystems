############################################################################
# pulse_shapes_symbo.py – symbolic pulse shapes
# ---------------------------------------------------------------------------
# This file contains symbolic definitions of pulse shapes using SymPy.
# These can be used with the DriveTerm class from drive_symbo.py.
############################################################################
from __future__ import annotations

import sympy as sp
from typing import Dict, Tuple, List, Optional, Set
from dataclasses import dataclass, field
import inspect

# Common symbols used across pulse shapes
t = sp.symbols('t', real=True)
amp = sp.symbols('amp', real=True)
t_start = sp.symbols('t_start', real=True)
t_duration = sp.symbols('t_duration', real=True)
t_rise = sp.symbols('t_rise', real=True)
t_square = sp.symbols('t_square', real=True)
t_stop = sp.symbols('t_stop', real=True)
amp_correction = sp.symbols('amp_correction', real=True)
how_many_sigma = sp.symbols('how_many_sigma', real=True)
stoke = sp.symbols('stoke', real=True)
delta1 = sp.symbols('delta1', real=True)
delta2 = sp.symbols('delta2', real=True)
phi = sp.symbols('phi', real=True)
normalize = sp.symbols('normalize', real=True)

@dataclass
class PulseParameters:
    """Container for pulse parameters with documentation and validation.
    
    This class ensures that:
    1. All required parameters are actually used in the expression
    2. All optional parameters with defaults are actually used in the expression
    3. No extra parameters are provided that aren't used
    4. Clear error messages are provided when validation fails
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
        # Get all symbols used in the expression
        used_symbols = {str(sym) for sym in expr.free_symbols}
        
        # Check required parameters
        missing_required = set(self.required_params) - used_symbols
        if missing_required:
            raise ValueError(
                f"Required parameters {missing_required} are not used in the expression. "
                f"Either remove them from required_params or add them to the expression."
            )
        
        # Check optional parameters
        unused_optional = set(self.optional_params.keys()) - used_symbols
        if unused_optional:
            raise ValueError(
                f"Optional parameters {unused_optional} are not used in the expression. "
                f"Either remove them from optional_params or add them to the expression."
            )
        
        # Check for extra symbols not documented
        extra_symbols = used_symbols - set(self.required_params) - set(self.optional_params.keys())
        if extra_symbols:
            raise ValueError(
                f"Symbols {extra_symbols} are used in the expression but not documented. "
                f"Either add them to required_params or optional_params, or remove them from the expression."
            )

def validate_pulse_definition(func):
    """Decorator to validate pulse definitions.
    
    This decorator ensures that:
    1. The function returns a tuple of (expr, defaults, params)
    2. The params match the expression
    3. The defaults match the optional_params
    """
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if not isinstance(result, tuple) or len(result) != 3:
            raise ValueError(
                f"Pulse definition {func.__name__} must return (expr, defaults, params). "
                f"Got {type(result)} with length {len(result) if isinstance(result, tuple) else 'N/A'}"
            )
        
        expr, defaults, params = result
        if not isinstance(params, PulseParameters):
            raise ValueError(
                f"Pulse definition {func.__name__} must return PulseParameters as third element. "
                f"Got {type(params)}"
            )
        
        # Validate params against expression
        params.validate_against_expr(expr)
        
        # Validate defaults match optional_params
        default_symbols = {str(sym) for sym in defaults.keys()}
        optional_symbols = set(params.optional_params.keys())
        if default_symbols != optional_symbols:
            raise ValueError(
                f"Defaults in {func.__name__} don't match optional_params. "
                f"Defaults: {default_symbols}, Optional: {optional_symbols}"
            )
        
        return result
    return wrapper

@validate_pulse_definition
def square_pulse_with_rise_fall() -> Tuple[sp.Expr, Dict[sp.Symbol, float], PulseParameters]:
    """Symbolic definition of a square pulse with rise and fall times."""
    t_fall_start = t_start + t_rise + t_square
    t_end = t_fall_start + t_rise
    
    # Define the three regions
    rise_window = sp.Piecewise(
        (1, (t >= t_start) & (t <= t_start + t_rise)),
        (0, True)
    )
    square_window = sp.Piecewise(
        (1, (t >= t_start + t_rise) & (t <= t_fall_start)),
        (0, True)
    )
    fall_window = sp.Piecewise(
        (1, (t >= t_fall_start) & (t <= t_end)),
        (0, True)
    )
    
    # Define the envelope
    rise_envelope = rise_window * sp.sin(sp.pi * (t - t_start) / (2 * t_rise)) ** 2
    square_envelope = square_window
    fall_envelope = fall_window * sp.sin(sp.pi * (t_end - t) / (2 * t_rise)) ** 2
    
    envelope = 2 * sp.pi * amp * (square_envelope + rise_envelope + fall_envelope)
    
    # Default values
    defaults = {
        t_start: 0,
        t_rise: 1e-13,
        t_square: 0
    }
    
    # Parameter documentation
    params = PulseParameters(
        required_params=['amp'],
        optional_params={
            't_start': 0,
            't_rise': 1e-13,
            't_square': 0
        },
        docstring="""Square pulse with rise and fall times.
        
        Required parameters:
        -------------------
        amp: float
            Amplitude of the pulse
            
        Optional parameters:
        -------------------
        t_start: float
            Start time of the pulse (default: 0)
        t_rise: float
            Rise time in seconds (default: 1e-13)
        t_square: float
            Duration of constant amplitude (default: 0)
        """
    )
    
    return envelope, defaults, params

@validate_pulse_definition
def sin_squared_pulse() -> Tuple[sp.Expr, Dict[sp.Symbol, float], PulseParameters]:
    """Symbolic definition of a sin-squared pulse."""
    t_end = t_start + t_duration
    
    inside_window = sp.Piecewise(
        (1, (t >= t_start) & (t <= t_end)),
        (0, True)
    )
    
    envelope = 2 * sp.pi * amp * inside_window * sp.sin(sp.pi * (t - t_start) / t_duration) ** 2
    
    # Default values
    defaults = {
        t_start: 0
    }
    
    # Parameter documentation
    params = PulseParameters(
        required_params=['amp', 't_duration'],
        optional_params={
            't_start': 0
        },
        docstring="""Sin-squared pulse.
        
        Required parameters:
        -------------------
        amp: float
            Amplitude of the pulse
        t_duration: float
            Duration of the pulse in seconds
            
        Optional parameters:
        -------------------
        t_start: float
            Start time of the pulse (default: 0)
        """
    )
    
    return envelope, defaults, params

@validate_pulse_definition
def sin_squared_DRAG() -> Tuple[sp.Expr, Dict[sp.Symbol, float], PulseParameters]:
    """Symbolic definition of a sin-squared DRAG pulse."""
    t_end = t_start + t_duration
    
    inside_window = sp.Piecewise(
        (1, (t >= t_start) & (t <= t_end)),
        (0, True)
    )
    
    envelope = inside_window * sp.sin(sp.pi * (t - t_start) / t_duration) ** 2
    envelope_derivative = inside_window * (sp.pi/t_duration) * sp.sin(2 * sp.pi * (t - t_start) / t_duration)
    
    complex_envelope = 2 * sp.pi * (amp * envelope - sp.I * amp_correction * envelope_derivative)
    
    # Default values
    defaults = {
        t_start: 0
    }
    
    # Parameter documentation
    params = PulseParameters(
        required_params=['amp', 'amp_correction', 't_duration'],
        optional_params={
            't_start': 0
        },
        docstring="""Sin-squared DRAG pulse.
        
        Required parameters:
        -------------------
        amp: float
            Amplitude of the pulse
        amp_correction: float
            Amplitude correction for DRAG
        t_duration: float
            Duration of the pulse in seconds
            
        Optional parameters:
        -------------------
        t_start: float
            Start time of the pulse (default: 0)
        """
    )
    
    return complex_envelope, defaults, params

@validate_pulse_definition
def gaussian_pulse() -> Tuple[sp.Expr, Dict[sp.Symbol, float], PulseParameters]:
    """Symbolic definition of a Gaussian pulse."""
    sigma = t_duration/how_many_sigma
    t_center = t_start + t_duration / 2
    
    gaussian = amp * sp.exp(-((t - t_center) ** 2) / (2 * sigma ** 2))
    
    t_end = t_start + t_duration
    inside_window = sp.Piecewise(
        (1, (t >= t_start) & (t <= t_end)),
        (0, True)
    )
    
    envelope = inside_window * gaussian
    
    # Add normalization if requested
    a = gaussian.subs(t, t_start)
    normalized_envelope = sp.Piecewise(
        ((envelope - a)/(1 - a), normalize),
        (envelope, True)
    )
    
    final_envelope = 2 * sp.pi * normalized_envelope
    
    # Default values
    defaults = {
        t_start: 0,
        how_many_sigma: 6,
        normalize: False
    }
    
    # Parameter documentation
    params = PulseParameters(
        required_params=['amp', 't_duration'],
        optional_params={
            't_start': 0,
            'how_many_sigma': 6,
            'normalize': False
        },
        docstring="""Gaussian pulse.
        
        Required parameters:
        -------------------
        amp: float
            Amplitude of the pulse
        t_duration: float
            Duration of the pulse in seconds
            
        Optional parameters:
        -------------------
        t_start: float
            Start time of the pulse (default: 0)
        how_many_sigma: float
            Number of standard deviations to include (default: 6)
        normalize: bool
            Whether to normalize the pulse (default: False)
        """
    )
    
    return final_envelope, defaults, params

@validate_pulse_definition
def gaussian_DRAG() -> Tuple[sp.Expr, Dict[sp.Symbol, float], PulseParameters]:
    """Symbolic definition of a Gaussian DRAG pulse."""
    sigma = t_duration/how_many_sigma
    t_center = t_start + t_duration / 2
    
    gaussian = amp * sp.exp(-((t - t_center) ** 2) / (2 * sigma ** 2))
    
    t_end = t_start + t_duration
    inside_window = sp.Piecewise(
        (1, (t >= t_start) & (t <= t_end)),
        (0, True)
    )
    
    envelope = inside_window * gaussian
    
    # Add normalization if requested
    a = gaussian.subs(t, t_start)
    normalized_envelope = sp.Piecewise(
        ((envelope - a)/(1 - a), normalize),
        (envelope, True)
    )
    
    drag_correction = 1 + sp.I * amp_correction * (-(t - t_center)/sigma**2)
    final_envelope = 2 * sp.pi * drag_correction * normalized_envelope
    
    # Default values
    defaults = {
        t_start: 0,
        how_many_sigma: 6,
        normalize: False
    }
    
    # Parameter documentation
    params = PulseParameters(
        required_params=['amp', 'amp_correction', 't_duration'],
        optional_params={
            't_start': 0,
            'how_many_sigma': 6,
            'normalize': False
        },
        docstring="""Gaussian DRAG pulse.
        
        Required parameters:
        -------------------
        amp: float
            Amplitude of the pulse
        amp_correction: float
            Amplitude correction for DRAG
        t_duration: float
            Duration of the pulse in seconds
            
        Optional parameters:
        -------------------
        t_start: float
            Start time of the pulse (default: 0)
        how_many_sigma: float
            Number of standard deviations to include (default: 6)
        normalize: bool
            Whether to normalize the pulse (default: False)
        """
    )
    
    return final_envelope, defaults, params

@validate_pulse_definition
def STIRAP_stoke() -> Tuple[sp.Expr, Dict[sp.Symbol, float], PulseParameters]:
    """Symbolic definition of a STIRAP stoke pulse."""
    lambda_val = 4
    tau_for_mono = (t_stop - t_start) / 6
    center = (t_stop - t_start) / 2 + t_start
    
    mono_increasing = 1 / (1 + sp.exp(-lambda_val * (t - center) / tau_for_mono))
    
    T0 = 2 * tau_for_mono
    hyper_Gaussian = sp.exp(-((t - center) / T0) ** (2*3))
    
    a = hyper_Gaussian.subs(t, t_start)
    
    envelope = (hyper_Gaussian - a)/(1-a) * sp.cos(sp.pi/2 * mono_increasing) * 2 * sp.pi * amp

    # Default values
    defaults = {
        t_start: 0
    }
    
    # Parameter documentation
    params = PulseParameters(
        required_params=['amp', 't_stop'],
        optional_params={
            't_start': 0
        },
        docstring="""STIRAP stoke pulse.
        
        Required parameters:
        -------------------
        amp: float
            Amplitude of the pulse
        t_stop: float
            Stop time of the pulse in seconds
            
        Optional parameters:
        -------------------
        t_start: float
            Start time of the pulse (default: 0)
        """
    )
    
    return envelope, defaults, params

@validate_pulse_definition
def STIRAP_pump() -> Tuple[sp.Expr, Dict[sp.Symbol, float], PulseParameters]:
    """Symbolic definition of a STIRAP pump pulse."""
    lambda_val = 4
    tau_for_mono = (t_stop - t_start) / 6
    center = (t_stop - t_start) / 2 + t_start
    
    mono_increasing = 1 / (1 + sp.exp(-lambda_val * (t - center) / tau_for_mono))
    
    T0 = 2 * tau_for_mono
    hyper_Gaussian = sp.exp(-((t - center) / T0) ** (2*3))
    
    a = hyper_Gaussian.subs(t, t_start)
    
    envelope = (hyper_Gaussian - a)/(1-a) * sp.sin(sp.pi/2 * mono_increasing) * 2 * sp.pi * amp
    
    # Default values
    defaults = {
        t_start: 0
    }
    
    # Parameter documentation
    params = PulseParameters(
        required_params=['amp', 't_stop'],
        optional_params={
            't_start': 0
        },
        docstring="""STIRAP pump pulse.
        
        Required parameters:
        -------------------
        amp: float
            Amplitude of the pulse
        t_stop: float
            Stop time of the pulse in seconds
            
        Optional parameters:
        -------------------
        t_start: float
            Start time of the pulse (default: 0)
        """
    )
    
    return envelope, defaults, params

@validate_pulse_definition
def Hyper_Gaussian_DRAG_STIRAP_stoke() -> Tuple[sp.Expr, Dict[sp.Symbol, float], PulseParameters]:
    """Symbolic definition of a Hyper-Gaussian DRAG STIRAP stoke pulse."""
    lambda_val = 4
    tau_mono = (t_stop - t_start) / 6
    center = (t_stop + t_start) / 2 + t_start
    
    def mono(t):
        return 1 / (1 + sp.exp(-lambda_val * (t - center) / tau_mono))
    
    T0 = 2 * tau_mono
    hyperG = sp.exp(-((t - center) / T0) ** 6)
    d_hyperG = hyperG * (-6) * ((t - center) / T0) ** 5 / T0
    
    a0 = hyperG.subs(t, t_start)
    
    envelope = (hyperG - a0) / (1 - a0) * sp.cos(sp.pi/2 * mono(t))
    d_envelope = (d_hyperG / (1 - a0) - (hyperG - a0) / (1 - a0)**2 * d_hyperG.subs(t, t_start)) * sp.cos(sp.pi/2 * mono(t)) + \
                 (hyperG - a0)/(1 - a0) * (-sp.pi/2) * sp.sin(sp.pi/2 * mono(t)) * (lambda_val / tau_mono) * mono(t) * (1 - mono(t))
    
    complex_envelope = 2 * sp.pi * (amp * envelope - sp.I * amp_correction * d_envelope)
    
    # Default values
    defaults = {
        t_start: 0
    }
    
    # Parameter documentation
    params = PulseParameters(
        required_params=['amp', 'amp_correction', 't_stop'],
        optional_params={
            't_start': 0
        },
        docstring="""Hyper-Gaussian DRAG STIRAP stoke pulse.
        
        Required parameters:
        -------------------
        amp: float
            Amplitude of the pulse
        amp_correction: float
            Amplitude correction for DRAG
        t_stop: float
            Stop time of the pulse in seconds
            
        Optional parameters:
        -------------------
        t_start: float
            Start time of the pulse (default: 0)
        """
    )
    
    return complex_envelope, defaults, params

@validate_pulse_definition
def Hyper_Gaussian_DRAG_STIRAP_pump() -> Tuple[sp.Expr, Dict[sp.Symbol, float], PulseParameters]:
    """Symbolic definition of a Hyper-Gaussian DRAG STIRAP pump pulse."""
    lambda_val = 4
    tau_mono = (t_stop - t_start) / 6
    center = (t_stop + t_start) / 2 + t_start
    
    def mono(t):
        return 1 / (1 + sp.exp(-lambda_val * (t - center) / tau_mono))
    
    T0 = 2 * tau_mono
    hyperG = sp.exp(-((t - center) / T0) ** 6)
    d_hyperG = hyperG * (-6) * ((t - center) / T0) ** 5 / T0
    
    a0 = hyperG.subs(t, t_start)
    
    envelope = (hyperG - a0) / (1 - a0) * sp.sin(sp.pi/2 * mono(t))
    d_envelope = (d_hyperG / (1 - a0) - (hyperG - a0) / (1 - a0)**2 * d_hyperG.subs(t, t_start)) * sp.sin(sp.pi/2 * mono(t)) + \
                 (hyperG - a0)/(1 - a0) * (sp.pi/2) * sp.cos(sp.pi/2 * mono(t)) * (lambda_val / tau_mono) * mono(t) * (1 - mono(t))
    
    complex_envelope = 2 * sp.pi * (amp * envelope - sp.I * amp_correction * d_envelope)
    
    # Default values
    defaults = {
        t_start: 0
    }
    
    # Parameter documentation
    params = PulseParameters(
        required_params=['amp', 'amp_correction', 't_stop'],
        optional_params={
            't_start': 0
        },
        docstring="""Hyper-Gaussian DRAG STIRAP pump pulse.
        
        Required parameters:
        -------------------
        amp: float
            Amplitude of the pulse
        amp_correction: float
            Amplitude correction for DRAG
        t_stop: float
            Stop time of the pulse in seconds
            
        Optional parameters:
        -------------------
        t_start: float
            Start time of the pulse (default: 0)
        """
    )
    
    return complex_envelope, defaults, params

@validate_pulse_definition
def sin_squared_recursive_DRAG() -> Tuple[sp.Expr, Dict[sp.Symbol, float], PulseParameters]:
    """Symbolic definition of a recursive DRAG pulse that punches out two notches."""
    # Get DRAG coefficients
    """Symbolic definition of recursive DRAG coefficients.
    
    Formula (Li et al. (2024), Appendix D):
        κ₂ = - A / (Δ₁ Δ₂)
        κ₁ =   A (Δ₁ + Δ₂) / (Δ₁ Δ₂)
    """
    kappa2 = -amp / (delta1 * delta2)
    kappa1 = amp * (delta1 + delta2) / (delta1 * delta2)
    
    # Define time variables
    t_end = t_start + t_duration
    two_pi_over_T = 2 * sp.pi / t_duration
    pi_over_T = sp.pi / t_duration
    
    # ---- rectangular window ----
    inside = sp.Piecewise(
        (1, (t >= t_start) & (t <= t_end)),
        (0, True)
    )
    
    # ---- sin² envelope and its first two derivatives ----
    envelope = inside * sp.sin(sp.pi * (t - t_start) / t_duration) ** 2
    
    # 1st derivative: (π/T)·sin(2πτ) with τ = (t-t₀)/T
    envelope_d1 = inside * pi_over_T * sp.sin(two_pi_over_T * (t - t_start))
    
    # 2nd derivative: (2π/T)² · (cos(2πτ) – 1)
    envelope_d2 = inside * 2 * (pi_over_T ** 2) * (sp.cos(two_pi_over_T * (t - t_start)) - 1)
    
    # Complex baseband envelope
    baseband = amp * envelope + sp.I * kappa1 * envelope_d1 + kappa2 * envelope_d2
    
    # Default values
    defaults = {
        t_start: 0
    }
    
    # Parameter documentation
    params = PulseParameters(
        required_params=['amp', 'delta1', 'delta2', 't_duration'],
        optional_params={
            't_start': 0
        },
        docstring="""Recursive DRAG pulse that punches out two notches.
        
        Required parameters:
        -------------------
        amp: float
            Real in-phase amplitude A
        delta1: float
            Detuning Δ₁ (rad/s) of 1st unwanted transition
            (use ±2π·MHz etc.; sign does not matter)
        delta2: float
            Detuning Δ₂ (rad/s) of 2nd unwanted transition
            (use ±2π·MHz etc.; sign does not matter)
        t_duration: float
            Pulse length in seconds
            
        Optional parameters:
        -------------------
        t_start: float
            Left edge of pulse window (default: 0)
        """
    )
    
    return baseband, defaults, params 