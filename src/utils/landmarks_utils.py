import numpy as np
import open3d as o3d

def gaussian_landmarks_homogeneous(centers, stddevs, num_points):
    landmarks = []
    for center, stddev in zip(centers, stddevs):
        x = np.random.normal(center[0], stddev[0], num_points)
        y = np.random.normal(center[1], stddev[1], num_points)
        z = np.random.normal(center[2], stddev[2], num_points)
        ones = np.ones(num_points)
        landmarks.append(np.column_stack((x, y, z, ones)))
    return np.vstack(landmarks)

def load_pcd(file_path, voxel_size=0.01):
    pcd = o3d.io.read_point_cloud(file_path)
    downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return downpcd