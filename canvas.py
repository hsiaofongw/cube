from os import WEXITED
import numpy as np
from helper import Helper
from math import tan
     
class Canvas:
    
    # create a camera and a canvas, 
    # a canvas is a 2-D thing use to paint 3-D objects
    def __init__(
        self, 
        camera: np.ndarray,
        view_direction: np.ndarray,
        canvas_distance: float,
    ):
        self.camera = camera
        view_direction = view_direction/np.linalg.norm(view_direction)
        self.view_direction = view_direction
        self.canvas_distance = canvas_distance
        
        self.canvas_center = camera + canvas_distance * view_direction
    
    # define the horizon for canvas, horizon should not parallel to the norm of canvas
    def set_horizon(self, horizon: np.ndarray) -> None:

        projection = self.view_direction  * np.inner(
            self.view_direction, horizon
        ) / np.inner(self.view_direction, self.view_direction)

        horizon = horizon - projection
        self.dir_x = horizon / np.linalg.norm(horizon)

        self.dir_y = np.cross(self.dir_x, self.view_direction)
        self.dir_y = self.dir_y / np.linalg.norm(self.dir_y)
        
    # get locations of each canvas pixel 
    def get_pixel_points(
        self, 
        angles_h: np.ndarray,
        angles_v: np.ndarray,
        width_pixels: np.ndarray,
        height_pixels: np.ndarray
    ) -> np.ndarray:

        left_arm = tan(angles_h[0]) * self.canvas_distance * self.dir_x
        right_arm = tan(angles_h[1]) * self.canvas_distance * self.dir_x

        top_arm = tan(angles_v[1]) * self.canvas_distance * self.dir_y
        bottom_arm = tan(angles_v[0]) * self.canvas_distance * self.dir_y

        baby_step_x = (right_arm - left_arm) / width_pixels
        baby_step_y = (bottom_arm - top_arm) / height_pixels

        baby_step_x = np.atleast_2d(baby_step_x)
        baby_step_y = np.atleast_2d(baby_step_y)

        x_steps, y_steps = np.meshgrid(
            np.arange(width_pixels),
            np.arange(height_pixels)
        )

        x_steps = x_steps.flatten(order='C')
        y_steps = y_steps.flatten(order='C')

        x_steps = np.atleast_2d(x_steps).T
        y_steps = np.atleast_2d(y_steps).T

        pixel_origin = self.canvas_center + left_arm + top_arm
        pixels = pixel_origin + x_steps * baby_step_x + y_steps * baby_step_y

        return pixels
        