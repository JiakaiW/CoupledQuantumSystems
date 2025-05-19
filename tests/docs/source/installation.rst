Installation
============

You can install CoupledQuantumSystems using pip:

.. code-block:: bash

    pip install CoupledQuantumSystems

For development installation:

.. code-block:: bash

    git clone https://github.com/JiakaiW/CoupledQuantumSystems.git
    cd CoupledQuantumSystems
    pip install -e .

Dependencies
-----------

The package requires the following dependencies:

- Python >= 3.7
- scipy >= 1.12.0
- numpy >= 1.26.4
- qutip >= 4.7.5
- scqubits >= 4.0.0
- loky
- bidict
- dynamiqs
- nevergrad
- rich

Optional dependencies for JAX support:

- jax[cuda12] 