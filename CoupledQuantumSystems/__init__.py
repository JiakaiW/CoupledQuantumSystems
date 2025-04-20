"""CoupledQuantumSystems: A Python package for simulating and optimizing coupled quantum systems.

This package provides tools for:
- Simulating the dynamics of coupled quantum systems
- Optimizing system parameters for desired quantum states
- Analyzing quantum system properties
- Visualizing quantum system dynamics
- Parallel computation of quantum evolutions

Example:
    >>> from CoupledQuantumSystems import run_optimization_with_progress
    >>> # Initialize your quantum system
    >>> # Run optimization
    >>> result = run_optimization_with_progress(system, target_state)
"""

from .optimize import (
    OptimizationProgress,
    evaluate_candidate,
    run_optimization_with_progress
)

__version__ = '0.3'
__author__ = 'Jiakai Wang'
__email__ = 'jwang2648@wisc.edu'

__all__ = [
    'OptimizationProgress',
    'evaluate_candidate',
    'run_optimization_with_progress'
]