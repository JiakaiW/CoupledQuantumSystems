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
from typing import Callable, Dict, Optional
import numpy as np
import qutip
import sympy as sp
from qiskit_dynamics import Signal

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

def _lambdify_both(expr: sp.Expr):
    """Return (env_np, env_jx|None) lambdas."""
    t = sp.symbols("t", real=True)
    env_np = sp.lambdify(t, expr, modules="numpy")
    if JAX_AVAILABLE:
        env_jx = sp.lambdify(t, expr, modules="jax")
    else:
        env_jx = None
    return env_np, env_jx

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

    # internal lambdas
    _env_np: Optional[Callable] = field(init=False, default=None)
    _env_jx: Optional[Callable] = field(init=False, default=None)

    # ------------------------------------------------------------------
    def __post_init__(self):
        if self.symbolic_expr is not None:
            # substitute parameters → numeric expr
            t = sp.symbols("t")
            expr = self.symbolic_expr.subs(self.symbolic_params)
            self._env_np, self._env_jx = _lambdify_both(expr)
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
        cos = jnp.cos; two_pi = 2*np.pi; ω = self.modulation_freq; ϕ = self.phi
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
            if self._env_jx is None:
                raise RuntimeError("SymPy→JAX lambdify failed; install JAX")
            env = self._env_jx
        else:
            env = lambda t: self.pulse_shape_func(t, self.pulse_shape_args, math=jnp)
        return Signal(
            envelope=env,
            carrier_freq=self.modulation_freq,
            phase=self.phi,
            name=self.pulse_id,
            jax_enable=True,
        )
