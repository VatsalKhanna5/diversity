import numpy as np


def rotation_matrix_2d(theta_rad: float) -> np.ndarray:
    return np.array(
        [[np.cos(theta_rad), -np.sin(theta_rad)], [np.sin(theta_rad), np.cos(theta_rad)]],
        dtype=float,
    )


def rotate_iq(symbols: np.ndarray, theta_rad: float) -> np.ndarray:
    iq = np.column_stack([np.real(symbols), np.imag(symbols)])
    rot = iq @ rotation_matrix_2d(theta_rad).T
    return rot[:, 0] + 1j * rot[:, 1]


def rotation_matrix_3x3(theta_rad: float) -> np.ndarray:
    # Orthonormal 3D rotation using two planar rotations.
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    r12 = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    phi = theta_rad / 2.0
    cp = np.cos(phi)
    sp = np.sin(phi)
    r23 = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    return r23 @ r12
