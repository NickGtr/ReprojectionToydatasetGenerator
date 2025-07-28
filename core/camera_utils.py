import numpy as np

def cam_to_world_inversion(cam_positions):
    """Get the extrinsics matrices from the camera positions

    Args:
        cam_positions (ndarray of shape (cam_nb, 3, 4)) : Positions and orientations of the cameras in the world frame.

    Returns:
        extrinsics (ndarray of shape (cam_nb, 3, 4)) : Extrinsics matrices used to express landmarks in the camera frame.
    """
    
    extrinsics = np.zeros((cam_positions.shape[0], 3, 4), dtype=np.float64)
    extrinsics[:, :, :3] = cam_positions[:, :, :3].transpose(0, 2, 1)
    extrinsics[:, :, 3] = - np.einsum('ijk,ik->ij', extrinsics[:, :, :3], cam_positions[:, :, 3])
    return extrinsics

def get_projection_matrices(intrinsics, extrinsics) -> np.ndarray:
    """Compute P = K[R|t] for each camera.

    Args:
        intrinsics (ndarray(n, 3, 3) or ndarray(3,3)): If a single intrinsic matrix : used throughtout the cameras
        extrinsics (ndarray(n, 3, 4)

    Returns:
        projection_matrices (np.ndarray(n, 3, 4)): K[R|t]
    """
    if intrinsics.ndim == 3:
        if intrinsics.shape[0] != extrinsics.shape[0]:
            raise TypeError("Mismatching shape between intrinsics and extrinsics")
        projection_matrices = np.einsum('nij,njk->nik', intrinsics, extrinsics)
    else:
        projection_matrices = np.einsum('ij,njk->nik', intrinsics, extrinsics)
    return projection_matrices