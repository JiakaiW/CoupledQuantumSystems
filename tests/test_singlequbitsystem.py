import unittest
import numpy as np
import qutip
import scqubits

from CoupledQuantumSystems.systems.singlequbitsystems import gfIFQ

class TestSingleQubitSystem(unittest.TestCase):
    def setUp(self):
        # Basic parameters for a fluxonium qubit
        self.EJ = 5.0  # Josephson energy in GHz
        self.EC = 1.0  # Charging energy in GHz
        self.EL = 0.5  # Inductive energy in GHz
        self.flux = 0.0  # External flux
        self.truncated_dim = 3  # Number of energy levels to consider

    def test_imports(self):
        """Test that all required imports are working."""
        # If we get here, all imports worked
        self.assertTrue(True)

    def test_fluxonium_initialization(self):
        """Test that we can create a fluxonium qubit instance."""
        qubit = gfIFQ(
            EJ=self.EJ,
            EC=self.EC,
            EL=self.EL,
            flux=self.flux,
            truncated_dim=self.truncated_dim
        )
        
        # Verify basic attributes
        self.assertEqual(qubit.truncated_dim, self.truncated_dim)
        self.assertIsInstance(qubit.evals, np.ndarray)
        self.assertEqual(len(qubit.evals), self.truncated_dim)
        self.assertIsInstance(qubit.diag_hamiltonian, qutip.Qobj)

if __name__ == '__main__':
    unittest.main() 