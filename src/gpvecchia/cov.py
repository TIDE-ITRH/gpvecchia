import numpy as np
from numba import njit



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
    alpha, beta, gamma = rotation_params
    quaternion = generate_quaternion(alpha, beta, gamma)
    
    # Get rotation matrix from quaternion
    rotation_matrix = quaternion_to_rotation_matrix(quaternion)
    
    # Apply rotation
    scaled_coords = np.dot(rotation_matrix, coords.T)
        
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