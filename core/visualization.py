import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


class VisualizationMixin:
    @staticmethod
    def create_camera_object(scale=1, cam_to_world=None):
        # Camera body (box)
        body = o3d.geometry.TriangleMesh.create_box(width=2*scale, height=scale, depth=scale)
        body.translate([-scale, -0.5*scale, -0.5*scale])
        body.paint_uniform_color([0.1, 0.1, 0.1])  # dark gray

        # Camera lens (cone)
        lens = o3d.geometry.TriangleMesh.create_cone(radius=0.5*scale, height=2*scale)
        lens.rotate(o3d.geometry.get_rotation_matrix_from_xyz([0, -np.pi/2, 0]))
        lens.translate([1.8 * scale, 0, 0])
        lens.paint_uniform_color([0.1, 0.1, 0.1])  # dark gray

        # Combine
        camera = body + lens
        camera.rotate(o3d.geometry.get_rotation_matrix_from_xyz([0, - np.pi / 2, 0]))
        camera.translate(-camera.get_center())
        camera.compute_vertex_normals()

        if cam_to_world is not None:
            cam_to_world_h = np.eye(4)
            cam_to_world_h[:3, :4] = cam_to_world
            camera.transform(cam_to_world_h)
        return camera
    
    def visualize_cameras_2d(self, plane='xy', scale=0.1):
        """
        Visualizes camera positions, orientations, and o3D landmark points in 2D.
        
        Parameters:

        - plane: str, 'xy', 'xz', or 'yz' to choose the projection plane.
        - scale: float, length of the direction arrow.
        """
        assert plane in ['xy', 'xz', 'yz'], "Plane must be 'xy', 'xz', or 'yz'"
        
        fig, ax = plt.subplots()
        ax.set_aspect('equal')

        for i, pose in enumerate(self.cam_to_world):
            R = pose[:, :3]
            t = pose[:, 3]

            if plane == 'xy':
                pos = t[[0, 1]]
                direction = R[[0, 1], 2]
            elif plane == 'xz':
                pos = t[[0, 2]]
                direction = R[[0, 2], 2]
            elif plane == 'yz':
                pos = t[[1, 2]]
                direction = R[[1, 2], 2]

            ax.plot(pos[0], pos[1], 'bo')
            ax.arrow(pos[0], pos[1],
                    scale * direction[0], scale * direction[1],
                    head_width=scale * 0.3, head_length=scale * 0.5, fc='r', ec='r')

            # Add index label slightly above the camera position
            ax.text(pos[0], pos[1] + scale * 0.5, str(i), color='blue', fontsize=9, ha='center')

        if plane == 'xy':
            points = self.landmarks[:, [0, 1]]
        elif plane == 'xz':
            points = self.landmarks[:, [0, 2]]
        elif plane == 'yz':
            points = self.landmarks[:, [1, 2]]
            
        ax.plot(points[:, 0], points[:, 1], 'k.', label='Landmarks')

        ax.set_title(f"Camera_positions and landmarks in {plane.upper()} plane")
        ax.grid(True)
        plt.xlabel(plane[0].upper())
        plt.ylabel(plane[1].upper())
        ax.legend()
        plt.show()

    def visualize_3D(self):
        to_display = [self.pcd]
        for cam in self.cam_to_world:
            to_display.append(self.create_camera_object(1, cam))
        o3d.visualization.draw_geometries(to_display)

    def visualize_projections(self, show_which_cam=None, sphere_size = 10, fig_size = 5, show_grid = False):
        """
        Visualizes 2D projections from multiple camera views.

        Parameters:
        - show_which_cam: list of int or None
            Indices of the cameras to show. If None, all cameras will be shown.
        """
        if self.observations is None:
            raise TypeError("No projections have been computed yet")

        if show_which_cam is None:
            show_which_cam = list(range(len(self.observations)))

        widths = self.sensor_sizes[:,0]
        heights = self.sensor_sizes[:,1]
        xlim = np.column_stack((-widths / 2, widths / 2))
        ylim = np.column_stack((-heights / 2, heights / 2))

        n_views = len(show_which_cam)
        fig, axes = plt.subplots(1, n_views, figsize=(fig_size * n_views, fig_size))

        if n_views == 1:
            axes = [axes]

        for ax, cam_idx in zip(axes, show_which_cam):
            proj = self.observations[cam_idx, :, :]
            if proj.shape[1] != 2:
                raise ValueError(f"Projection at index {cam_idx} must be of shape (N, 2), got {proj.shape}")

            if self.colors is not None:
                if len(self.colors) != proj.shape[0]:
                    raise TypeError("Mismatch of color matrices and projection matrix")
                ax.scatter(proj[:, 0], proj[:, 1], s=sphere_size, c=self.colors)
            else:
                ax.scatter(proj[:, 0], proj[:, 1], s=sphere_size)
            ax.set_xlim(xlim[cam_idx])
            ax.set_ylim(ylim[cam_idx][::-1])  # Invert y-axis so origin is at the center-top
            ax.set_title(f"Camera {cam_idx}")
            ax.set_aspect('equal')
            ax.grid(show_grid)

        plt.tight_layout()
        plt.show()