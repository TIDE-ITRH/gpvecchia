import numpy as np
from numba import njit
import math


def euclidean_distance(x1, x2):
    return np.sqrt(((x1[:,None,:] - x2[None,:,:])**2).sum(axis=-1))

@njit(cache=True)
def matern32(d,l):
    """
    Non scaled [var 1] Matern 3/2 that takes a distance martrix as an input (i.e. not a vector of coordinates as above)
    
    Parameters
    ==========
    l: Matern length scale
    d: distance matrix
    """

    fac1 = 3*d**2
    fac2 = np.sqrt(fac1)
    return (1 + fac2/l)*np.exp(-fac2/l)


@njit(cache=True)
def cosine(d, l):
    """Non-scaled cosine base function"""
    return np.cos(2*np.pi*np.abs(d)/l)



@njit(cache=True)
def matern32_1d(X, Xp, covparams):
    eta, l = covparams
    d = 0.0
    for i in range(len(X)):
        d += (X[i] - Xp[i])**2
    return eta**2. * matern32(d**0.5, l)







### Rotation stuff


@njit(cache=True)
def matern_general(d, eta, nu):
    # Precompute constants
    sqrt_2nu = math.sqrt(2 * nu)
    gamma_factor = math.gamma(nu)
    scale_factor = (eta**2) * (2**(1 - nu)) / gamma_factor

    # Compute the scaled distance
    cff1 = sqrt_2nu * abs(d)

    # Handle the case where distance is zero
    if cff1 == 0:
        return eta**2

    # Compute the Matern covariance
    return scale_factor * (cff1**nu) * math.kv(nu, cff1)



# @njit(cache=True, fastmath=True)
def generate_quaternion(alpha, beta, gamma):
    """
    Generates a quaternion from the reparameterized angles alpha, beta, gamma.
    
    Parameters:
    alpha, beta, gamma: angles that parameterize the quaternion.
    
    Returns:
    Quaternion [q0, q1, q2, q3] (normalized).
    """
    q0 = np.cos(alpha)
    q1 = np.sin(alpha) * np.cos(beta)
    q2 = np.sin(alpha) * np.sin(beta) * np.cos(gamma)
    q3 = np.sin(alpha) * np.sin(beta) * np.sin(gamma)
    
    return [q0, q1, q2, q3]

# @njit(cache=True, fastmath=True)
def quaternion_to_rotation_matrix(q):
    """
    Converts a quaternion to a 3x3 rotation matrix.
    
    Parameters:
    q: list or array of size 4 [q0, q1, q2, q3] (w, x, y, z)
    
    Returns:
    3x3 rotation matrix
    """
    q0, q1, q2, q3 = q
    R = np.array([
        [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
    ])
    return R

# @njit(cache=True)
def scale_coordinates(coords, length_scales):
    """
    Scales coordinates by a length scale matrix.
    
    Parameters:
    coords: 1D array or list, [x, y, z]
    length_scales: 1D array or list, the length scales [l_x, l_y, l_z]
    
    Returns:
    Scaled coordinates.
    """
    # scaling_matrix = np.linalg.inv(np.diag(length_scales))
    scaling_matrix = np.diag(1/np.array(length_scales))
    scaled_coords = np.dot(scaling_matrix, coords)
    return scaled_coords

def transform_coordinates(coords, rotation_params, length_scales):
    """
    Scales and rotates the coordinates using reparameterized quaternion and length scales.
    
    Parameters:
    coords: 1D array or list, [x, y, z]
    rotation_params (alpha, beta, gamma): angles that define the quaternion rotation.
    length_scales: 1D array or list, the length scales [l_x, l_y, l_z]
    
    Returns:
    Transformed coordinates.
    """
    # Generate quaternion from reparameterized angles
    if rotation_params is not None:
        assert len(rotation_params) == 3, "rotation_params must be a list or array of length 3"
        alpha, beta, gamma = rotation_params
        quaternion = generate_quaternion(alpha, beta, gamma)
        
        # Get rotation matrix from quaternion
        rotation_matrix = quaternion_to_rotation_matrix(quaternion)
        
        # Apply rotation
        scaled_coords = np.dot(rotation_matrix, coords.T)
    else:
        scaled_coords = coords.T
        
    # Scale coordinates
    rotated_scaled_coords = scale_coordinates(scaled_coords, length_scales)
    
    return rotated_scaled_coords.T

# def build_scaling_matrix(length_scales):
#     """
#     Build an n x n covariance matrix where the diagonal contains the length scales
#     and the off-diagonals contain the covariances between dimensions.

#     Parameters
#     ----------
#     length_scales: list or array of length scales (diagonal elements)
#     covariances: list or array of covariances (off-diagonal elements)

#     Returns
#     -------
#     covariance_matrix: n x n numpy array
#         The resulting covariance matrix.
#     """
#     n = len(length_scales)

#     # Create an empty matrix
#     covariance_matrix = np.zeros((n, n))

#     # Fill the diagonal with length scales
#     for i in range(n):
#         covariance_matrix[i, i] = length_scales[i]

    # # Fill the off-diagonal elements with covariances
    # if covariances is None:
    #     return covariance_matrix
    
    # else:
    #     k = 0
    #     for i in range(n):
    #         for j in range(i + 1, n):
    #             covariance_matrix[i, j] = covariances[k]
    #             covariance_matrix[j, i] = covariances[k]
    #             k += 1

    #     return covariance_matrix