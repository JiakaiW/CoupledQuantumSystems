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

from .systems import *
from .dynamics import *
from .utils import *


__version__ = '0.3'
__author__ = 'Jiakai Wang'
__email__ = 'jwang2648@wisc.edu'

# It's generally not recommended to modify __all__ when using 'import *'
# as it becomes hard to maintain. If specific exports are needed,
# consider importing them explicitly and adding to __all__.
# For now, the '*' imports make symbols available directly.
# The symbols previously listed here (from optimize.py) are now available
# through 'from .utils import *'.
__all__ = []

# If you want to extend __all__ with names from the new submodules,
# you'd need to list them explicitly or inspect the modules.
# For example:
# from .systems import QuantumSystem # assuming QuantumSystem is in systems
# __all__.append('QuantumSystem')