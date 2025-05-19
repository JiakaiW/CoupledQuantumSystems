import numpy as np
import scqubits
from functools import partial


############################################################################
#
#
# Functions for single qubit c_ops estimations
#
#
############################################################################


hbar = 1/(2*np.pi)
kB = 8.617333262e-5  # eV K−1
hbar_in_eVs = 6.5821e-16  # eV s
temp_in_mK = 20


############################################################################
# T_phi
############################################################################

def old_T_phi(second_order_derivative, one_over_f_flux_noise_amplitude, first_order_derivative= 0 ):  # eqn (13) of Peter Groszkowski et al 2018 New J. Phys. 20 043053
    omega_uv = 20 * 2 * np.pi  # 20GHz
    omega_ir = 1e-9 * 2 * np.pi  # 1Hz
    t = 1e6  # 1ms
    A = one_over_f_flux_noise_amplitude  # in unit of Phi0

    first_order_part = 2 * A**2 * first_order_derivative**2
    first_order_part *= np.abs(np.log(omega_ir * t))

    second_order_part = 2 * A**4 * second_order_derivative**2  # Phi0^4 GHZ^2 / Phi0^4
    second_order_part *= (np.log(omega_uv / omega_ir)**2 + 2 * np.log(omega_ir * t)**2)  # GHZ^2
    return (first_order_part + second_order_part)  **(-1/2)  # ns

# This is probably wrong without the cut off terms
def T_phi(second_order_derivative, one_over_f_flux_noise_amplitude, first_order_derivative= 0 ):
    A = one_over_f_flux_noise_amplitude  # in unit of Phi0

    first_order_part = A * np.abs(first_order_derivative)
    second_order_part =  A**2 * np.abs(second_order_derivative)
    return 1/(first_order_part + second_order_part)

def second_order_derivative(f, x0, rtol=1e-3, atol=1e-4, max_iter=20):
    h = 1e-3
    derivative_old = 0.0
    # print('\n')
    for i in range(max_iter):
        h /= 2
        derivative_new = (f(x0 + h) - 2 * f(x0) + f(x0 - h)) / h**2
        # print(derivative_new)
        if np.abs(derivative_new - derivative_old) < rtol*np.abs(derivative_old):
            return derivative_new
        derivative_old = derivative_new
    raise ValueError("Convergence not reached within the maximum number of iterations")

def first_order_derivative(f, x0, rtol=1e-3, atol=1e-4, max_iter=20):
    h = 1e-3
    derivative_old = 0.0
    # print('\n')
    for i in range(max_iter):
        h /= 2
        derivative_new = (f(x0 + h) - f(x0 - h)) / (2 * h)
        # print(derivative_new)
        if np.abs(derivative_new - derivative_old) < rtol * np.abs(derivative_old):
            return derivative_new
        derivative_old = derivative_new
    raise ValueError("Convergence not reached within the maximum number of iterations")


############################################################################
# T_1
############################################################################


# def diel_spectral_density(omega, EC,temp_in_mK = 20 ,tangent_ref = 1e-5):
#     beta = 1 / (kB * temp_in_mK * 1e-3)  # 1/eV

#     coth_arg = beta * hbar_in_eVs * np.abs(omega) / 2  # s GHZ
#     coth_arg *= 1e9  # dimensionless
#     return_val = np.where(omega < 0, 
#                           1/2 * np.abs( 1 / np.tanh(coth_arg) - 1) , 
#                           1/2 * np.abs( 1 / np.tanh(coth_arg) + 1) )

#     omega_ref = 2*np.pi *6 # GHz
#     epsilon = 0.15
#     Q_cap = 1/(  2* tangent_ref * np.abs(omega/omega_ref)**epsilon ) 

#     return_val *= hbar * np.abs(omega)**2   / (4 * EC * Q_cap)  # GHZ^2/GHZ = GHZ
#     return return_val


def diel_spectral_density(omega, EC,temp_in_mK = 20 ,Q_cap = 1e5):
    beta = 1 / (kB * temp_in_mK * 1e-3)  # 1/eV

    x = beta * hbar_in_eVs * omega # s GHZ
    x *= 1e9  # dimensionless
    return_val = 1/2 * np.abs( 1 / np.tanh(x/ 2) + 1) 

    return_val *= hbar * np.abs(omega)**2   / (4 * EC * Q_cap)  # GHZ^2/GHZ = GHZ
    return return_val

def one_over_f_spectral_density(omega, EL,one_over_f_flux_noise_amplitude ):
    return_val = 2 * np.pi # dimensionless
    return_val *= (EL / hbar)**2  # GHz^2
    return_val *= one_over_f_flux_noise_amplitude**2  # GHz^2
    return_val /= omega  # GHz
    return np.abs(return_val)

############################################################################
# Fluxonium
############################################################################


def get_fluxonium_frequency_with_2pi(flux,EJ,EC,EL,i,j,truncated_dim=30):
    qbt = scqubits.Fluxonium(EJ = EJ,EC = EC,EL =EL, cutoff = 110,flux = flux,truncated_dim=truncated_dim)
    vals = qbt.eigenvals(qbt.truncated_dim)
    return np.abs(vals[j]-vals[i])*2*np.pi
