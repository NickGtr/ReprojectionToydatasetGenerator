import numpy as np

class ExportationMixin:

    def save_to_NPZfile(self, filename):
        np.savez_compressed(
            filename,
            landmarks=self.landmarks,
            intrinsics=self.intrinsics,
            extrinsics=self.extrinsics,
            sensor_sizes=self.sensor_sizes,
            colors=self.colors if self.colors is not None else np.array([]),
            observations=self.observations if self.observations is not None else np.array([])
        )
        print(f"Saved ToyModel to {filename}")
        
    def export_observations_and_ground_truths(self, path):
        cam_nb = self.cam_nb
        lm_nb = self.lm_nb
        obs_nb = 0

        for lm_idx in range(lm_nb):
            for cam_idx in range(cam_nb):
                if not (np.isnan(self.observations[cam_idx][lm_idx][0]) | np.isnan(self.observations[cam_idx][lm_idx][1])):
                    obs_nb += 1

        with open(path, 'w') as f:
            f.write(f"{cam_nb} {lm_nb} {obs_nb}" + "\n")
            for lm_idx in range(lm_nb):
                for cam_idx in range(cam_nb):
                    projection = self.observations[cam_idx][lm_idx]
                    if not (np.isnan(projection[0]) | np.isnan(projection[1])):
                        f.write(f"{cam_idx} {lm_idx} {projection[0]} {projection[1]}" + "\n")
            
            for intrinsic, extrinsic, sensor_size in zip(self.intrinsics, self.extrinsics, self.sensor_sizes):
                space_matrix = intrinsic @ extrinsic
                f.write(" ".join(str(coordinate) for coordinate in space_matrix.flatten()) + "\n")
                f.write(f"{sensor_size[0]} {sensor_size[1]}" + "\n")

            for landmark in self.landmarks:
                cartesian_landmark = landmark[:3] / landmark[3]
                f.write(f"{cartesian_landmark[0]} {cartesian_landmark[1]} {cartesian_landmark[2]}" + "\n")

    def export_observations_with_random_initialization(self, path, init_extrinsics, init_cartesian_landmarks):
        cam_nb = self.cam_nb
        lm_nb = self.lm_nb
        obs_nb = 0

        init_extrinsics = np.asarray(init_extrinsics)
        init_cartesian_landmarks = np.asarray(init_cartesian_landmarks)

        if init_extrinsics.shape != (cam_nb, 3, 4):
            raise TypeError(f"extrinsics should be of shape ({cam_nb}, 3, 4). Got {init_extrinsics.shape}")
        if init_cartesian_landmarks.shape != (lm_nb, 3):
            raise TypeError(f"landmarks should be of shape ({lm_nb}, 3, 4). Got {init_cartesian_landmarks.shape}")

        for lm_idx in range(lm_nb):
            for cam_idx in range(cam_nb):
                if not (np.isnan(self.observations[cam_idx][lm_idx][0]) | np.isnan(self.observations[cam_idx][lm_idx][1])):
                    obs_nb += 1

        with open(path, 'w') as f:
            f.write(f"{cam_nb} {lm_nb} {obs_nb}" + "\n")
            for lm_idx in range(lm_nb):
                for cam_idx in range(cam_nb):
                    projection = self.observations[cam_idx][lm_idx]
                    if not (np.isnan(projection[0]) | np.isnan(projection[1])):
                        f.write(f"{cam_idx} {lm_idx} {projection[0]} {projection[1]}" + "\n")
            
            for extrinsic, sensor_size in zip(init_extrinsics, self.sensor_sizes):
                f.write(" ".join(str(coordinate) for coordinate in extrinsic.flatten()) + "\n")
                f.write(f"{sensor_size[0]} {sensor_size[1]}" + "\n")

            for cartesian_landmark in init_cartesian_landmarks:
                f.write(f"{cartesian_landmark[0]} {cartesian_landmark[1]} {cartesian_landmark[2]}" + "\n")