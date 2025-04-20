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

The package provides efficient CPU multiprocessing for QuTiP solvers.

Basic Usage
~~~~~~~~~~

.. code-block:: python

    from CoupledQuantumSystems import run_parallel_ODEsolve_and_post_process_jobs_with_different_systems
    from loky import get_reusable_executor

    # Prepare your systems and parameters
    systems = [system1, system2, system3]
    kwargs_list = [
        {'y0': y0_1, 'tlist': tlist_1, 'drive_terms': drive_terms_1},
        {'y0': y0_2, 'tlist': tlist_2, 'drive_terms': drive_terms_2},
        {'y0': y0_3, 'tlist': tlist_3, 'drive_terms': drive_terms_3}
    ]

    # Run parallel simulations
    results = run_parallel_ODEsolve_and_post_process_jobs_with_different_systems(
        list_of_systems=systems,
        list_of_kwargs=kwargs_list,
        max_workers=4,  # Number of CPU cores to use
        store_states=True
    )

Performance Tips
~~~~~~~~~~~~~~~

- Use ``loky`` for better process management
- Adjust ``max_workers`` based on your CPU capabilities
- Consider memory usage when storing states

GPU Acceleration with dynamiqs
----------------------------

The package supports GPU-accelerated simulations using dynamiqs with checkpointing capabilities.

Basic Usage
~~~~~~~~~~

.. code-block:: python

    from CoupledQuantumSystems import DynamiqsSolver

    # Initialize the solver
    solver = DynamiqsSolver(
        system=your_system,
        checkpoint_interval=1000,  # Save state every 1000 steps
        checkpoint_dir='./checkpoints'
    )

    # Run simulation
    result = solver.solve(
        tlist=tlist,
        y0=y0,
        drive_terms=drive_terms,
        use_gpu=True  # Enable GPU acceleration
    )

    # Load from checkpoint if needed
    result = solver.load_from_checkpoint('checkpoint_1000.npz')

Checkpointing Features
~~~~~~~~~~~~~~~~~~~~

- Automatic checkpointing at specified intervals
- Manual checkpoint saving and loading
- GPU memory management
- State vector compression options

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    solver = DynamiqsSolver(
        system=your_system,
        checkpoint_interval=1000,
        checkpoint_dir='./checkpoints',
        compression_level=3,  # Higher compression, slower save/load
        gpu_memory_fraction=0.8,  # Limit GPU memory usage
        use_mixed_precision=True  # Use mixed precision for better performance
    )

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~

- Adjust checkpoint interval based on simulation duration
- Use compression for large state vectors
- Monitor GPU memory usage
- Consider mixed precision for faster computation

Example Notebooks
---------------

The package includes several example notebooks demonstrating various features:

- Basic usage examples
- Advanced system configurations
- Performance optimizations
- Visualization examples 