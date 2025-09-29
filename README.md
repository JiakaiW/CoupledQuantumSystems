# CoupledQuantumSystems-The Pytorch lightening for hamiltonian simulation


# Installation
CPU plain version:
```
pip3 install git+https://github.com/JiakaiW/CoupledQuantumSystems
```
dynamiqs (via JAX extra)
```
pip install "CoupledQuantumSystems[jax] @ git+https://github.com/JiakaiW/CoupledQuantumSystems"
```
qiskit + qiskit-dynamics
```
pip install "CoupledQuantumSystems[qiskit] @ git+https://github.com/JiakaiW/CoupledQuantumSystems"
```
both
```
pip install "CoupledQuantumSystems[jax,qiskit] @ git+https://github.com/JiakaiW/CoupledQuantumSystems"
```

to use it with dynamiqs, I recommand using the requirements.lock (without those strict versions I sometimes see very low level errors from gpu dynamiqs.mesolve)
'''
pip install --upgrade pip
pip install -r requirements.lock \
            -c requirements.lock      # lock file wins
pip install .[jax]                    # project deps obey the constraints
'''

# Package usecase

Quality of life improvements for quantum simulations,
- GPU simulation chekcpointing
- Product basis - dressed basis assignment and fast conversion
- Pulse library that supports qutip, dynamiqs, ScalableSymbolicPulse with qiskit-dynamics
- a lot more

# Backend support of QuTip, Dynamiqs, Qiskit-dynamics:
## Specify time dynamics with `DriveTerm` and `DriveTermSymbolic`
For time-dependent simulation, we can often write the Hamiltonian as $H = H_{static}+\sum_i H_{drive}^i,$ and typically the drive part can be written as a frequency modulation times a pulse envelope times the driven operator $H_{drive}^i = A(t)\sin(\omega_d t+\phi)\hat{O}$. 

In CoupledQuantumSystems, this pattern is reflected in the `ODEsolve_and_post_process` method, where `static_hamiltonian` captures $H_{static}$, and `drive_terms` specifies $\sum_i H_{drive}^i$. 

The tranditional `DriveTerm` class uses `driven_op`, `pulse_shape_func`, `pulse_shape_args` as in the typical qutip usage. A library of `pulse_shape_func` is available in `CoupledQuantumSystems/dynamics/drive.py`.

The second `DriveTermSymbolic` class is used to interface with Qiskit-dynamics. It can be initialized in the tranditional way using `legacy_pulse_shape_func` and `legacy_pulse_shape_args`, it can also be initialized with `pulse_type` and `symbolic_params`, a library of symbolic pulse shapes are availble in `CoupledQuantumSystems/dynamics/pulse_shapes_symbo.py`.

### Interfacing with backends with `ODEsolve_and_post_process`
The two types of drive terms mentioned above can be used in `ODEsolve_and_post_process`, and the actual backend solve can be specified with the `method` argument.

1. QuTip solvers (qutip.sesolve, qutip.mesolve) are conventional and easy to use.
2. Dynamiqs solvers are not interfaced through `ODEsolve_and_post_process`. GPU simulation with Dynamiqs are available with Checkpointing via `CheckpointingJob` (see `CoupledQuantumSystems/utils/checkpoint.py` and `Examples/demo_dynamiqs_gpu_mesolve_checkpoint.py`).
3. Qiskit-dynamics which implements rotating frame which speeds up simulation is available in `ODEsolve_and_post_process` via `method ='qiskit_dynamics'`, and the `rotating_frame`, `rwa_cutoff_freq`, `rwa_carrier_freqs` parameters. Note that a conventional qutip-style `DriveTerm` is still compatible with `method ='qiskit_dynamics'` via the `DriveTerm.envelope_to_qiskit_Signal()` method. (I forgot why I made it so compatible. So what's the meaning of implementing `DriveTermSymbolic`? LOL.)


# Usecase example: auto-diff GPU piece-wise-constant 2-qubit gate optimization

![gpu_meme](assets/gpu_meme.jpg)



















