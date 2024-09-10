# CoupledQuantumSystems
```
pip3 install git+https://github.com/JiakaiW/CoupledQuantumSystems
```

![workflow of quantum simulation using this package](assets/CoupledQuantumSystems.png)

### Note that this is primarily for personal use. Structure is not quite organized. (may call it early alpha)

## Features:
1. Abstract and facilitate the common workflows in dealing with coupled systems (particularly fluxonium)
2. Heavily uses numpy to deal with manipulations of qutip.qobj


## TODO:
1. implement product state to dressed state assignment by adiabatically turning on the coupling.
2. Add calling of jax solvers (dynamiqs) once their machine-precision solvers are out