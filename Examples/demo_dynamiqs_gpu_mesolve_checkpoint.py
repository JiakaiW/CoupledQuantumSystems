import numpy as np
from CoupledQuantumSystems import TransmonOscillatorSystem
import scqubits
import qutip
from CoupledQuantumSystems import DriveTerm
import dynamiqs as dq
from dynamiqs import QArray
from CoupledQuantumSystems import DriveTerm, square_pulse_with_rise_fall_envelope
from CoupledQuantumSystems import CheckpointingJob
import pickle


dq.set_progress_meter(True)
dq.set_device('gpu')
dq.set_precision('double')

def get_system():
    ncut = 110
    N18_qubit_params = {"EJ": 26, "EC": 0.3, "Er": 10}
    qbt_params = N18_qubit_params

    ol_trunc = 60
    max_ql=30
    max_ol=80
    ng=0.0
    qbt_lvls_to_keep = [1,9,14]

    system  =  TransmonOscillatorSystem(
                qbt =scqubits.Transmon(EJ=qbt_params["EJ"], EC=qbt_params["EC"], ng=ng,ncut=ncut,truncated_dim=max_ql),
                osc = scqubits.Oscillator(E_osc=qbt_params["Er"], truncated_dim=max_ol, l_osc=1.0),
                g_strength = 1.2,
                products_to_keep=[[ql, ol] for ql in qbt_lvls_to_keep for ol in range(ol_trunc)],
                )
    # qbt_lvls_to_keep = [1,9,14]
    # ol_trunc = 60
    # system.set_new_product_to_keep([[ql, ol] for ql in qbt_lvls_to_keep for ol in range(ol_trunc)])
    # system.set_new_operators_after_setting_new_product_to_keep()
    tlist =np.linspace(0,20,201)
    drive_terms=[DriveTerm(
        driven_op=system.driven_operator,
        pulse_shape_func= square_pulse_with_rise_fall_envelope,
        pulse_shape_args={"amp":0.1,"t_square":100},
        modulation_freq=system.evals[system.product_to_dressed[1,1]]-system.evals[system.product_to_dressed[1,0]]
    )]
    static_hamiltonian = system.diag_hamiltonian
    e_ops = [qutip.ket2dm(system.truncate_function(qutip.basis(max_ql*max_ol,system.product_to_dressed[(ql,ol)]))) for ql in qbt_lvls_to_keep for ol in range(ol_trunc)]+[system.truncate_function(system.a)]+[system.truncate_function(system.a.dag()*system.a)]
    c_ops = [0.3* dq.asqarray(system.truncate_function(system.a))]
    rho0 = dq.asqarray(system.truncate_function(qutip.basis(max_ql*max_ol,system.product_to_dressed[(1,0)])))
    tsave = tlist
    jump_ops = c_ops
    exp_ops = e_ops
    method =  dq.method.Tsit5(max_steps=  int(1e9),rtol=1e-10,atol=1e-10)
    len_t_segment_per_chunk = 40
    return static_hamiltonian, drive_terms, rho0, tsave, jump_ops, exp_ops, method, len_t_segment_per_chunk

if __name__ == '__main__':
    job = CheckpointingJob(name=f"tmon_readout")
    if not job.system_loaded:
        static_hamiltonian, drive_terms, rho0, tsave, jump_ops, exp_ops, method, len_t_segment_per_chunk = get_system()
        job.set_system(static_hamiltonian=static_hamiltonian, drive_terms=drive_terms, rho0=rho0, tsave=tsave, jump_ops=jump_ops, exp_ops=exp_ops, method=method, len_t_segment_per_chunk=len_t_segment_per_chunk)
    job.run_segment(progress_meter=True)
