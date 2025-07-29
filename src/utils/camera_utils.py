import numpy as np
from scipy.spatial.transform import Rotation as Rot
import matplotlib.pyplot as plt
import open3d as o3d

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

def camera_positions_arccircle(radial_range = np.pi/4, radial_offset = 0, radius = 1, cam_nb = 10, center = [0, 0, 0], radius_noise_stddev = 0, translation_noise_stddev = 0, rotation_angle_noise_stddev = 0):
    """Create an arccircle of cameras pointing towards the center.
    BEWARE : it is not the extrinsics matrix, but the camera positions and orientations in the world frame

    Args:
        radial_range (float): the actual range of the arc in radians
        radial_offset (float): move the arc along the circle.
        radius (float)
        cam_nb (int)
        center (list): where the cameras point to.
        radius_noise_stddev (flaot): effective radius to center of each camera differ by a gaussian noise multiplicator
        translation_noise_stddev(float): camera positions differ from theoretical arccircle by a gaussian noise
        rotation_angle_noise_stddev(float): camera orientation differ from looking at the center by a gaussian noise on the angles


    Returns:
        cam_to_world (ndarray of shape (cam_nb, 3, 4)). The first three rows are the rotation matrix defining camera orientation,
                                                the last row is the position of the camera in the world frame in inhomogeneous coordinates.
    """
    center = np.array(center, dtype=np.float64)
    cam_to_world = np.zeros((cam_nb,3, 4), dtype=np.float64)
    angles = np.linspace(radial_offset-radial_range/2, radial_offset+radial_range/2, cam_nb)
    for i in range(cam_nb):
        theta = angles[i]
        R = np.array([
            [-np.sin(theta), 0, -np.cos(theta)],
            [np.cos(theta), 0, -np.sin(theta)],
            [0, -1, 0]
            ])
        if rotation_angle_noise_stddev > 0:
            random_axis = np.random.normal(size=3)
            random_axis /= np.linalg.norm(random_axis)
            random_angle = np.random.normal(scale=rotation_angle_noise_stddev)
            noise_rotation = Rot.from_rotvec(random_axis * random_angle).as_matrix(
            )
            R =  noise_rotation @ R
        if radius_noise_stddev > 0:
            radius_noise = np.random.normal(scale=radius_noise_stddev)
        else:
            radius_noise = 0
        t = np.array([
            radius*(1 + radius_noise)*np.cos(theta) + center[0],
            radius*(1 + radius_noise)*np.sin(theta) + center[1],
            center[2],
        ])
        if translation_noise_stddev > 0:
            t += np.random.normal(scale=translation_noise_stddev, size=3)

        cam_to_world[i] = np.hstack([R, t.reshape(3, 1)])
    return cam_to_world

def lognormal_focal_length_generator(cam_nb, mean, sigma, show_distribution):
    
    focal_lengths = np.random.lognormal(mean, sigma, cam_nb)
    intrinsics = np.array([[
        [focal_lengths[i], 0, 0],
        [0, focal_lengths[i], 0],
        [0, 0, 1]
    ] for i in range(cam_nb)])

    if show_distribution:
        plt.hist(focal_lengths, bins=20, edgecolor='black')
        plt.title('Lognormal Distribution of Focal Lengths')
        plt.xlabel('Focal Length')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()
    return intrinsics

def get_visualiser_camera_parameters(pcd):
    """Navigate to a camera position of you liking and then close the window to get the camera position of the camera you were looking at the scene with

    Args:
        pcd : Point Cloud

    Returns:
        ndarray(3,3), ndarray(3,4) : Camera intrinsics & Camera position and orientation
    """

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    vis.run()

    view_control = vis.get_view_control()

    camera_parameters = view_control.convert_to_pinhole_camera_parameters()
    cam_to_world = np.linalg.inv(camera_parameters.extrinsic)[:3,:]
    intrinsics = camera_parameters.intrinsic.intrinsic_matrix
    print("Camera position and orientation (camera_to_world):")
    print(cam_to_world)

    print("Camera Intrinsic matrix:")
    print(intrinsics)

    camera_translation = cam_to_world[:, 3]
    print("Camera position:", camera_translation)

    vis.destroy_window()

    return intrinsics, cam_to_world