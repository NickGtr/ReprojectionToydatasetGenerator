import numpy as np
import open3d as o3d

from .projection import ProjectionMixin
from .exportation import ExportationMixin
from .visualization import VisualizationMixin

class ToyModel(ProjectionMixin, ExportationMixin, VisualizationMixin):
    def __init__(self, landmarks, intrinsics, sensor_sizes = [36,24], extrinsics = None, cam_to_world = None, colors = None):
        """Stores all the necessary information of a ToyModel for SfM testing

        Args:
            landmarks (ArrayLike(m,4)) : should be in homogeneous form
            intrinsics (ArrayLike(n, 3, 3) or ArrayLike(3, 3)) : intrinsic matrices. If one is given, it used across all cameras
            sensor_sizes (ArrayLike(n, 2) or ArrayLike(2)) : Size of the sensors : must use the same scale as intrinsic matrices. Defauts to full-frame (36, 24)
            extrinsics (ArrayLike(n, 3, 4), optional): world_to_cam pose matrices. If not defined, cam_to_world must be.
            cam_to_world (_type_, optional) : _description_. cam_to_world pose matrices. If not defined, world_to_cam must be.
            colors (ArrayLike(m, 3), optional) : The colors of the different landmarks

        """
        self.landmarks = np.asarray(landmarks)
        assert self.landmarks.shape[1] == 4, "Landmarks must be in homogeneous coordinates"
        self.intrinsics = np.asarray(intrinsics)
        self.sensor_sizes = np.asarray(sensor_sizes)
        self.colors = colors
        self.observations = None

        if extrinsics is None:
            if cam_to_world is None:
                raise ValueError("At least one of cam_to_world or extrinsics must be declared")
            else:

                self.cam_to_world = np.asarray(cam_to_world)
                self.extrinsics = np.zeros((self.cam_to_world.shape[0], 3, 4))
                self.extrinsics[:, :, :3] = self.cam_to_world[:, :, :3].transpose(0, 2, 1)
                self.extrinsics[:, :, 3] = - np.einsum('ijk,ik->ij', self.extrinsics[:, :, :3], self.cam_to_world[:, :, 3])
                
        else:
            self.extrinsics = np.asarray(extrinsics)
            self.cam_to_world = np.zeros((self.extrinsics.shape[0], 3, 4))
            self.cam_to_world[:, :, :3] = self.extrinsics[:, :, :3].transpose(0, 2, 1)
            self.cam_to_world[:, :, 3] = - np.einsum('ijk,ik->ij', self.cam_to_world[:, :, :3], self.extrinsics[:, :, 3])

        self.lm_nb = self.landmarks.shape[0]
        self.cam_nb = self.extrinsics.shape[0]

        if self.intrinsics.ndim == 2:
            self.intrinsics = np.tile(self.intrinsics, (self.cam_nb, 1, 1))
        elif self.intrinsics.shape[0] != self.cam_nb:
            raise TypeError("Intrinsics and Extrinsics aren't of the same shape")
        
        if self.sensor_sizes.ndim == 1:
            self.sensor_sizes = np.tile(self.sensor_sizes, (self.cam_nb, 1))
        elif self.sensor_sizes.shape[0] != self.cam_nb:
            raise TypeError("Intrinsics and SensorSizes aren't of the same shape")

        cartesian_landmarks = self.landmarks[:, :3] / self.landmarks[:, 3][:, np.newaxis]
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(cartesian_landmarks)

        if self.colors is not None:
            self.colors = np.asarray(self.colors)
            if self.colors.shape[0] != self.lm_nb:
                raise TypeError("Number of colors should be the same as the number of landmarks")
            self.pcd.colors = o3d.utility.Vector3dVector(self.colors)

        self.projection_matrices = np.einsum('nij,njk->nik', self.intrinsics, self.extrinsics)


class ToyModel_From_PointCloud(ToyModel):
    def __init__(self, pcd, intrinsics, sensor_sizes = [36,24], extrinsics = None, cam_to_world = None):
        inhomogeneous_landmarks = np.asarray(pcd.points)
        ones = np.ones((inhomogeneous_landmarks.shape[0],1))
        homogeneous_landmarks = np.hstack((inhomogeneous_landmarks, ones))
        
        super().__init__(
            landmarks=homogeneous_landmarks,
            intrinsics=intrinsics,
            sensor_sizes=sensor_sizes,
            extrinsics=extrinsics,
            cam_to_world=cam_to_world,
            colors=pcd.colors
        )

        self.pcd = pcd
    
class ToyModel_From_SavedNPZ(ToyModel):
    def __init__(self, filename):
        data = np.load(filename, allow_pickle=True)
        landmarks = data['landmarks']
        intrinsics = data['intrinsics']
        extrinsics = data['extrinsics']
        sensor_sizes = data['sensor_sizes']
        colors = data['colors'] if data['colors'].size > 0 else None
        observations = data['observations'] if data['observations'].size > 0 else None

        super().__init__(
            landmarks=landmarks,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            sensor_sizes=sensor_sizes,
            colors=colors
        )

        self.observations = observations