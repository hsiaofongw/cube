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
        horizon = Helper.vertical_line(
            self.canvas_center, 
            self.canvas_center + self.view_direction,
            self.canvas_center + horizon
        )
        self.base_i = horizon / np.linalg.norm(horizon)
        self.view_direction = self.view_direction / np.linalg.norm(self.view_direction)
        self.base_j = np.cross(self.base_i, self.view_direction)
        
    # get locations of each canvas pixel 
    def get_pixel_points(
        self, 
        angles_h: np.ndarray,
        angles_v: np.ndarray,
        width_pixels: np.ndarray,
        height_pixels: np.ndarray
    ) -> np.ndarray:

        left_arm = tan(angles_h[0]) * self.canvas_distance * self.base_i
        right_arm = tan(angles_h[1]) * self.canvas_distance * self.base_i
        top_arm = tan(angles_v[1]) * self.canvas_distance * self.base_j
        bottom_arm = tan(angles_v[0]) * self.canvas_distance * self.base_j

        baby_step_j = (right_arm - left_arm) / width_pixels
        baby_step_i = (bottom_arm - top_arm) / height_pixels

        v_steps, h_steps = np.meshgrid(np.arange(height_pixels), np.arange(width_pixels))
        n_pixels = width_pixels * height_pixels
        h_steps = np.reshape(h_steps, newshape=(n_pixels, 1,))
        v_steps = np.reshape(v_steps, newshape=(n_pixels, 1,))

        pixel_origin = self.canvas_center + left_arm + top_arm
        pixels = pixel_origin + h_steps * baby_step_i + v_steps * baby_step_j

        return pixels
        