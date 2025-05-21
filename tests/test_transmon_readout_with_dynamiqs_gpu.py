import pytest

try:
    import jax
    import jax.numpy as jnp
    import dynamiqs as dq
    from dynamiqs import QArray
except ImportError:
    pytest.skip("Skipping GPU test module: JAX or dynamiqs not installed.", allow_module_level=True)

import numpy as np
import matplotlib.pyplot as plt
from CoupledQuantumSystems import TransmonOscillatorSystem
from CoupledQuantumSystems import DriveTerm, square_pulse_with_rise_fall_envelope
import scqubits
import qutip

dq.set_progress_meter(True)

try:
    gpu_devices = [d for d in jax.devices() if d.platform.lower() == 'gpu']
    if not gpu_devices:
        pytest.skip("Skipping GPU test module: No GPU device found by JAX.", allow_module_level=True)
    dq.set_device('gpu') # Original set_device call, now conditional
except Exception as e:
    pytest.skip(f"Skipping GPU test module: Error during JAX GPU check or dq.set_device - {e}", allow_module_level=True)

# dq.set_precision('double') # This commented line is preserved

if __name__ == '__main__':
    tlist = np.linspace(0,5,21)
    max_ql = 3
    max_ol = 4

    ncut = 110

    tmon_EJ = 30
    tmon_EC = 0.2
    tmon_ng = 0.3
    Er = 6
    g = 0.3
    kappa = 0.03
    qbt = scqubits.Transmon(EJ=tmon_EJ, EC=tmon_EC, ng=tmon_ng,ncut=ncut,truncated_dim=max_ql)
    osc = scqubits.Oscillator(E_osc=Er, truncated_dim=max_ol, l_osc=1.0)

    system  =  TransmonOscillatorSystem(
                qbt = qbt,
                osc = osc,
                g_strength = g,
                kappa=kappa,
                products_to_keep=[[ql, ol] for ql in range(max_ql) for ol in range(max_ol)],
                )
    diag_hamiltonian = system.diag_hamiltonian
    w_d =  system.evals[system.product_to_dressed[0,1]]-system.evals[system.product_to_dressed[0,0]]
    drive_terms = [
                    DriveTerm(
                    driven_op=system.driven_operator,
                    pulse_shape_func=square_pulse_with_rise_fall_envelope,
                    pulse_id='drive',  # Stoke is the first pulse, pump is the second
                    modulation_freq = w_d,
                    pulse_shape_args={
                        'amp': 0.1,
                        't_square':1e9
                    },
                ),
            ]

    def _H(t:float)->QArray:
        _H = dq.asqarray(diag_hamiltonian)
        for term in drive_terms:
            _H += dq.asqarray(term.driven_op)* term.jax_wrapper()(t, term.get_pulse_shape_args_with_id())
        return _H

    H =  dq.timecallable(_H, discontinuity_ts = None)
    e_ops = [qutip.ket2dm(qutip.basis(max_ql*max_ol,system.product_to_dressed[ql,ol])) for ql in range(max_ql) for ol in range(max_ol)]
    c_ops = [0.022* dq.asqarray(system.a)]

    result = dq.mesolve(
        H = H,
        rho0 = dq.asqarray(qutip.basis(max_ql*max_ol,0)),
        tsave = tlist,
        jump_ops = c_ops,
        exp_ops = [dq.asqarray(e_op) for e_op in e_ops],
        method = dq.method.Tsit5(max_steps=  int(1e9),rtol=1e-6,atol=1e-6)
        )
