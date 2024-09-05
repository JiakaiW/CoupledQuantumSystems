from CoupledQuantumSystems.qobj_manip import dressed_to_product_vectorized

import numpy as np


def test_dressed_to_product_vectorized():

    def unvectorized_code(product_to_dressed, dressed_dm_data, sign_multiplier):
        subsystem_dims = [max(indexes) + 1 for indexes in zip(*product_to_dressed.keys())]
        rho_product = np.zeros((subsystem_dims * 2), dtype=complex)

        for product_state, dressed_index1 in product_to_dressed.items():
            for product_state2, dressed_index2 in product_to_dressed.items():
                element = dressed_dm_data[dressed_index1, dressed_index2] * sign_multiplier[dressed_index1] * sign_multiplier[dressed_index2]
                rho_product[product_state + product_state2] += element

        return rho_product

    # Mock data for 3 dimensions
    product_to_dressed = {
        (0, 0, 0): 0,
        (0, 0, 1): 1,
        (0, 1, 0): 2,
        (0, 1, 1): 3,
        (1, 0, 0): 4,
        (1, 0, 1): 5,
        (1, 1, 0): 6,
        (1, 1, 1): 7
    }
    
    dressed_dm_data = np.array([[1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j, 5 + 5j, 6 + 6j, 7 + 7j, 8 + 8j],
                                [2 + 2j, 5 + 5j, 6 + 6j, 7 + 7j, 8 + 8j, 9 + 9j, 10 + 10j, 11 + 11j],
                                [3 + 3j, 6 + 6j, 8 + 8j, 9 + 9j, 10 + 10j, 11 + 11j, 12 + 12j, 13 + 13j],
                                [4 + 4j, 7 + 7j, 9 + 9j, 10 + 10j, 11 + 11j, 12 + 12j, 13 + 13j, 14 + 14j],
                                [5 + 5j, 8 + 8j, 10 + 10j, 11 + 11j, 13 + 13j, 14 + 14j, 15 + 15j, 16 + 16j],
                                [6 + 6j, 9 + 9j, 11 + 11j, 12 + 12j, 14 + 14j, 15 + 15j, 16 + 16j, 17 + 17j],
                                [7 + 7j, 10 + 10j, 12 + 12j, 13 + 13j, 15 + 15j, 16 + 16j, 18 + 18j, 19 + 19j],
                                [8 + 8j, 11 + 11j, 13 + 13j, 14 + 14j, 16 + 16j, 17 + 17j, 19 + 19j, 20 + 20j]], dtype=complex)
    
    sign_multiplier = np.array([1, -1, 1, -1, 1, -1, 1, -1])

    # Run the generalized vectorized code
    vectorized_result = dressed_to_product_vectorized(product_to_dressed, dressed_dm_data, sign_multiplier)
    original_result = unvectorized_code(product_to_dressed, dressed_dm_data, sign_multiplier)

    assert np.allclose(original_result, vectorized_result), "The results do not match!"

