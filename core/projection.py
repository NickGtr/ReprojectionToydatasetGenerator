import numpy as np
import matplotlib.pyplot as plt

class ProjectionMixin:
    def project_no_clipping(self):
        """ Project landmarks onto the image plane of each camera.
            All projections are kept (sensor size is infinite at this point and there is complete transparency).

        Returns:
            projections (ndarray(n, m, 2)): Projections of the landmarks onto the image plane of each camera.
        """

        projections = np.zeros((self.cam_nb, self.lm_nb, 2), dtype=np.float64)
        for i in range(self.cam_nb):
            projection = np.einsum('ij,mj->mi', self.projection_matrices[i], self.landmarks)
            projections[i] = projection[:, :2] / projection[:, 2][:, np.newaxis]
        return projections
    
    def get_observations_Zbuffered(self, buffering_pixel_nb=(640, 480), buffer_tolerance = 0.05, show_lm_per_pixel_dist = []):
        """Project landmarks onto the image sensor of each camera. Only the observation on the sensors are kept.
            If 2 points land in the same "buffering_pixel" and are far away in the scene, we keep the closest one. A method inspired by z-buffering is used for that matter.

        Args:
            buffering_pixel_nb (tuple, optional): Number of pixels used for the Z-buffering. Defaults to (640, 480).
            buffer_tolerance (float, optional) : Tolerance ratio of depth within a pixel beyond which far-away points are eliminated. Defaults to 0.05.
            show_lm_per_pixel_dist (list of indices, optionale) : display the distributioon of the number of lm per pixel for the given camera indices after the hidden surface removal.
                                                                    Use it to calibrate the Z-buffering.

        Returns:
            projections (ndarrau(n, m, 2)) : list of observations, with NaN values when no observation is made.
        """

        landmarks_in_camera_coordinates = np.einsum('nij,mj->nmi', self.extrinsics, self.landmarks)
        projections = np.zeros((self.cam_nb, self.lm_nb, 2), dtype=np.float64)
        pixel_projections = np.zeros((self.cam_nb, self.lm_nb, 2), dtype=np.int16)
        for i in range(self.cam_nb):
            projection = np.einsum('ij,mj->mi', self.intrinsics[i], landmarks_in_camera_coordinates[i])
            projections[i] = projection[:, :2] / projection[:, 2][:, np.newaxis]

        # For each camera, we look on which buffer pixel the projection lands
        width_pixel, height_pixel = buffering_pixel_nb
        width_sensors = self.sensor_sizes[:, 0][:, np.newaxis]
        height_sensors =  self.sensor_sizes[:, 1][:, np.newaxis]

        scale_x = width_pixel / width_sensors
        scale_y = height_pixel / height_sensors

        x_pixel_array = (projections[:, :, 0] + width_sensors / 2) * scale_x
        y_pixel_array = (projections[:, :, 1] + height_sensors / 2) * scale_y

        x_pixel_array = np.round(x_pixel_array).astype(np.int16)
        y_pixel_array = np.round(y_pixel_array).astype(np.int16)

        # Place NaN wherever the projection lands outside of the sensor
        x_mask = (x_pixel_array >= 0) & (x_pixel_array < width_pixel)
        y_mask = (y_pixel_array >= 0) & (y_pixel_array < height_pixel)
        mask = x_mask & y_mask
        projections[~mask] = np.nan

        # Store the pixel coordinate of each landmark in each camera. Place -1 when the point is not valid
        pixel_projections[:, :, 0] = np.where(mask, x_pixel_array, -1)
        pixel_projections[:, :, 1] = np.where(mask, y_pixel_array,  -1)

        z_buffer = np.full((self.cam_nb, width_pixel, height_pixel), np.inf, dtype=np.float64)
        lm_per_pixel_count = np.full((self.cam_nb, width_pixel, height_pixel), 0, dtype=np.int16)

        for cam_idx in range(self.cam_nb):
            for lm_idx in range(self.lm_nb):
                x_pixel, y_pixel = pixel_projections[cam_idx, lm_idx]
                if x_pixel == -1 or y_pixel == -1:
                    continue
                z = landmarks_in_camera_coordinates[cam_idx, lm_idx, 2]
                if z < z_buffer[cam_idx, x_pixel, y_pixel]:
                    z_buffer[cam_idx, x_pixel, y_pixel] = z

        for cam_idx in range(self.cam_nb):
            for lm_idx in range(self.lm_nb):
                x_pixel, y_pixel = pixel_projections[cam_idx, lm_idx]
                if x_pixel == -1 or y_pixel == -1:
                    continue
                z = landmarks_in_camera_coordinates[cam_idx, lm_idx, 2]
                if z > (1 + buffer_tolerance) * z_buffer[cam_idx, x_pixel, y_pixel]:
                    projections[cam_idx][lm_idx][0] = np.nan
                    projections[cam_idx][lm_idx][1] = np.nan
                else:
                    lm_per_pixel_count[cam_idx, x_pixel, y_pixel] += 1


        if len(show_lm_per_pixel_dist) != 0:
            plt.figure(figsize=(10, 6))
            for cam_idx in show_lm_per_pixel_dist:
                flattened_count = lm_per_pixel_count[cam_idx].flatten()
                plt.hist(flattened_count,
                    bins=np.arange(flattened_count.max() + 2) - 0.5,
                    alpha=0.5,
                    label=f'Camera {cam_idx}',
                    histtype='stepfilled',
                    density=True)
            plt.xlabel("Number of landmark points per pixel")
            plt.ylabel("Number of pixels")
            plt.title("Distribution of Landmark Points per Pixel per Camera")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.ylim(0,0.2)
            plt.show()
        self.observations = projections
        return projections
    
    def remove_occlusions_and_downsample(self, buffering_pixel_nb=(640, 480), buffer_tolerance = 0.05, show_lm_per_pixel_dist = [], voxel_size=0.02):
        self.get_observations_Zbuffered(buffering_pixel_nb, buffer_tolerance, show_lm_per_pixel_dist)

        _ , _ , indices = self.pcd.voxel_down_sample_and_trace(
            voxel_size=voxel_size,
            min_bound=self.pcd.get_min_bound(),
            max_bound=self.pcd.get_max_bound()
)
        
        #We proceed this way to apply the same downsampling across all cameras because Z-buffering has been already computed
        post_down_sampling_indices = [i[0] for i in indices if len(i) > 0]

        print("Downsampling landmarks")
        self.remove_landmarks(post_down_sampling_indices)
        self.remove_unobserved_landmarks(minimum_observations=3)
        self.remove_unseeing_cameras(minimum_observations=1)
        
        return self.observations

    def remove_cameras(self, mask_or_indices):
        """Remove given cameras

        Args:
            mask_or_indices (ArrayLike of boolean(size cam_nb) or int indices): which cameras to keep
        """
        if mask_or_indices.dtype == bool:
            if mask_or_indices.shape[0] != self.cam_nb:
                raise TypeError(f"the boolean mask must be of size {self.cam_nb}. Got {len(mask_or_indices)}")
        
        if self.observations is not None:
            self.observations = self.observations[mask_or_indices, :, :]
            
        self.intrinsics = self.intrinsics[mask_or_indices, :, :]
        self.extrinsics = self.extrinsics[mask_or_indices, :, :]
        self.cam_to_world = self.cam_to_world[mask_or_indices, :, :]
        self.sensor_sizes = self.sensor_sizes[mask_or_indices, :]
        self.cam_nb = self.cam_to_world.shape[0]

        print(f"ToyDataset now contains {self.cam_nb} cameras")

    def remove_landmarks(self, mask_or_indices):
        """Remove given landmarks

        Args:
            mask_or_indices (ArrayLike of boolean(size lm_nb) or int indices) : which landmarks to keep
        """
        mask_or_indices = np.asarray(mask_or_indices)
        if mask_or_indices.dtype == bool:
            if mask_or_indices.shape[0] != self.lm_nb:
                raise TypeError(f"the boolean mask must be of size {self.lm_nb}. Got {len(mask_or_indices)}")
            else:
                indices = [idx for idx in range(self.lm_nb) if mask_or_indices[idx]]
        
        
        self.pcd = self.pcd.select_by_index(indices)

        if self.observations is not None:
            self.observations = self.observations[:, mask_or_indices, :]

        inhomogeneous_landmarks = np.asarray(self.pcd.points)
        if self.colors is not None:
            self.colors = self.pcd.colors
        ones = np.ones((inhomogeneous_landmarks.shape[0],1))
        landmarks_homogeneous = np.hstack((inhomogeneous_landmarks, ones))
        self.landmarks = landmarks_homogeneous
        self.lm_nb = self.landmarks.shape[0]

        print(f"ToyDataset now contains {self.lm_nb} landmarks")

    def remove_unobserved_landmarks(self, minimum_observations=3):
        """Landmarks observed less that minimum_observations are removed"""

        if self.observations is None:
            raise ValueError("No observations computed yet.")

        valid_mask = ~np.isnan(self.observations).any(axis=2)
        nb_observations_per_lm = np.sum(valid_mask, axis=0)
        print(nb_observations_per_lm)
        lm_mask = nb_observations_per_lm >= minimum_observations
        print(lm_mask)

        print(f"Removing landmarks observed less than {minimum_observations} times.")
        self.remove_landmarks(lm_mask)

    def remove_unseeing_cameras(self, minimum_observations = 1):
        """Cameras observing less than minimum_observations landmarks are removed"""

        if self.observations is None:
            raise ValueError("No observations computed yet.")
        
        valid_mask = ~np.isnan(self.observations).any(axis=2)
        nb_observations_per_cam = np.sum(valid_mask, axis=1)
        cam_mask = nb_observations_per_cam >= minimum_observations

        print(f"Removing cameras observing less than {minimum_observations} landmarks.")
        self.remove_cameras(cam_mask)

