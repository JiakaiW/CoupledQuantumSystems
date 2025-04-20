Usage
=====

Usage Guide
===========

This guide provides detailed documentation for the key features of CoupledQuantumSystems.

CoupledSystem Class
-----------------

The ``CoupledSystem`` class provides a powerful framework for handling coupled quantum systems, managing the mapping between product basis and energy eigenbasis.

Basic Usage
~~~~~~~~~~

.. code-block:: python

    from CoupledQuantumSystems import CoupledSystem, QubitResonatorSystem

    # Initialize a coupled system (e.g., qubit-resonator)
    system = QubitResonatorSystem(
        qbt=qubit_instance,  # Pre-initialized qubit
        osc=oscillator_instance,  # Pre-initialized oscillator
        g_strength=0.1,  # Coupling strength
        products_to_keep=['00', '01', '10', '11']  # Product states to keep
    )

    # Access the mapping between product and dressed states
    product_to_dressed = system.product_to_dressed
    dressed_to_product = system.dressed_to_product

    # Convert between bases
    product_state = system.convert_dressed_to_product_vectorized(dressed_state)
    dressed_state = system.convert_product_to_dressed_vectorized(product_state)

Advanced Features
~~~~~~~~~~~~~~~

- **State Truncation**: Use ``truncate_function`` to reduce the Hilbert space
- **State Padding**: Use ``pad_back_function`` to handle state vectors of different dimensions
- **Custom Mappings**: Override ``alternative_product_to_dressed`` for custom state mappings

CPU Multiprocessing with QuTiP
-----------------------------

The package provides efficient CPU multiprocessing for QuTiP solvers using the `run_qutip_mesolve_parrallel` method.

Basic Usage
~~~~~~~~~~

.. code-block:: python

    from CoupledQuantumSystems import QuantumSystem

    # Initialize your quantum system
    system = QuantumSystem()

    # Prepare your initial states and parameters
    initial_states = [qutip.basis(2, 0)]
    tlist = np.linspace(0, 10, 100)
    drive_terms = [DriveTerm(qutip.sigmax(), lambda t, args: np.sin(t))]

    # Run parallel simulations
    results = system.run_qutip_mesolve_parrallel(
        initial_states=initial_states,
        tlist=tlist,
        drive_terms=[drive_terms],
        e_ops=[qutip.sigmaz()]
    )

Performance Tips
~~~~~~~~~~~~~~~

- Use ``loky`` for better process management
- Adjust ``max_workers`` based on your CPU capabilities
- Consider memory usage when storing states

GPU Acceleration
----------------

Currently, the package does not support GPU acceleration with `DynamiqsSolver`. Please refer to the latest updates in the repository for future support.

Example Notebooks
---------------

The package includes several example notebooks demonstrating various features:

- Basic usage examples
- Advanced system configurations
- Performance optimizations
- Visualization examples 